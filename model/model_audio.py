import os
import warnings
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import AutoProcessor

from .model_minimind import *
from transformers import Qwen2AudioForConditionalGeneration, Qwen3ForCausalLM, Qwen3Config
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache

warnings.filterwarnings('ignore')

class ALMConfig(Qwen3Config):
    model_type = "minimind-a"

    def __init__(
        self,
        audio_special_token: Optional[str] = None,
        audio_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        # 先调用父类初始化，加载所有标准参数
        super().__init__(**kwargs)
        
        # 设置自己的额外参数，若没有传入则赋默认值
        self.audio_special_token = audio_special_token if audio_special_token is not None else '$' * 750
        self.audio_ids = audio_ids if audio_ids is not None else [69502] * 750


# inherit from language model
class MiniMindALM(Qwen3ForCausalLM):
    config_class = ALMConfig

    def __init__(self, params: ALMConfig = None, audio_size=1280, 
                 hidden_size=2048, audio_model_path="Qwen/Qwen2-Audio-7B-Instruct"):
        super().__init__(params)
        if not params: params = ALMConfig()
        self.params = params
        self.multi_modal_projector = nn.Linear(audio_size, hidden_size)

        qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(audio_model_path)
        self.processor = AutoProcessor.from_pretrained(audio_model_path)
        self.audio_tower = qwen_model.audio_tower
        
    @staticmethod
    def select_continuous_audio_mask(input_ids, audio_id, max_audio_tokens):
        batch_size, seq_len = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for b in range(batch_size):
            for idx in range(seq_len - max_audio_tokens + 1):
                window = input_ids[b, idx:idx + max_audio_tokens]
                if torch.all(window == audio_id).item():
                    mask[b, idx:idx + max_audio_tokens] = True
                    break
        return mask

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                input_features: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                feature_attention_mask: Optional[torch.Tensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        
        # get the input text embeddings
        hidden_states = self.model.embed_tokens(input_ids)
        
        # merge text and audio
        if input_features is not None and input_ids.shape[1] != 1:
            audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            batch_size, _, _, max_mel_seq_len = input_features.shape
            input_features = input_features.view(batch_size, -1, max_mel_seq_len)
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feat_lengths.expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand
            
            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device
            )
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask)
            selected_audio_feature = audio_outputs.last_hidden_state
            audio_features = self.multi_modal_projector(selected_audio_feature)

            # if we have consecutive audio tokens, then it means we expanded input_ids in processing
            audio_tokens = input_ids == self.params.audio_ids[0]
            legacy_processing = (audio_tokens[:, :-1] & audio_tokens[:, 1:]).sum() == 0
            
            if not legacy_processing:
                num_audios, max_audio_tokens, embed_dim = audio_features.shape
                audio_features_mask = torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None, :]
                audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
                
                audio_features_mask = audio_features_mask.squeeze(1)
                # audio_features = [audio_features[i][audio_features_mask[i]] for i in range(batch_size)]

                special_audio_mask = (input_ids == self.params.audio_ids[0]).to(hidden_states.device)
                if special_audio_mask.sum() > max_audio_tokens * batch_size:
                    # exists the same text token as audio token
                    special_audio_mask = self.select_continuous_audio_mask(input_ids, 
                                                                      self.params.audio_ids[0], 
                                                                      max_audio_tokens).to(hidden_states.device)
                
                audio_features = audio_features.to(hidden_states.device, hidden_states.dtype)
                audio_features_mask = audio_features_mask.to(hidden_states.device, attention_mask.dtype)
                attention_mask = attention_mask.to(hidden_states.device)
                
                # insert the padding mask of audio features in attention_mask
                attention_mask = attention_mask.masked_scatter(special_audio_mask, audio_features_mask)
                special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(hidden_states)    
                hidden_states = hidden_states.masked_scatter(special_audio_mask, audio_features)
            else:
                raise ValueError('Legacy processing audio tokens! Please insert <audio> tokens in the sequence.')

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # model_inputs = self.prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, cache_position)
        
        outputs: BaseModelOutputWithPast = super().forward(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # return {
        #     'last_hidden_state': outputs.hidden_states,
        #     'logits': outputs.logits
        # }

        return CausalLMOutputWithPast(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None
        )