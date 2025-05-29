import os
import io
import json
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pyarrow.parquet as pq
from transformers import AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AudioPretrainDataset(Dataset):
    def __init__(self, 
                 root,
                 tokenizer,
                 processor,
                 max_length=1024,
                 audio_special_token='$' * 750):
        super().__init__()
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url="train-clean-100", download=False)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.audio_token = audio_special_token
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.dataset)

    def _create_chat_prompt(self, text):
        transcription_prompts = [
            "Please transcribe the following audio:",
            "Could you provide a transcription of this audio?",
            "Transcribe the audio clip below:",
            "Kindly write down what you hear in the audio:",
            "What does the following audio say?",
            "Convert the following audio into text:",
            "Listen and transcribe the following audio:",
            "Please listen carefully and write the transcript of the audio below:",
            "I need the transcription of this audio sample:",
            "Here's an audio clip — what does it say?"
            ]
        prompt = random.choice(transcription_prompts)
        
        messages = [
            {"role": "user", "content": f"{prompt}\n{self.audio_token}"},
            {"role": "assistant", "content": text}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[index]

        prompt = self._create_chat_prompt(transcript)
        input_ids = self.tokenizer(prompt).input_ids
        origin_len = len(input_ids)
        input_ids = input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)
        
        padding_mask = [1] * origin_len + [0] * (self.max_length - origin_len)
        padding_mask = padding_mask[:self.max_length]

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        padding_mask = torch.tensor(padding_mask[1:], dtype=torch.long)

        input_features = self.processor.feature_extractor(
            waveform[0], # whisper处理一维的输入,而这里的waveform维度是[1, -1]
            sampling_rate=sample_rate,
            padding="max_length", 
            return_attention_mask=True, 
            return_tensors="pt"
        )
        feature_attention_mask = input_features.pop("attention_mask")

        return X, Y, loss_mask, feature_attention_mask, padding_mask, input_features['input_features']


class AudioSFTDataset(Dataset):
    def __init__(self, 
                 root_path, 
                 tokenizer, 
                 processor,
                 max_length=1024,
                 audio_special_token='$' * 750):
        super().__init__()
        self.root_path = root_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.audio_token = audio_special_token
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        self.samples = self.load_data(root_path)

    def load_data(self, root_path):
        root = Path(root_path)
        samples = []

        for file in root.rglob('*'):
            if file.suffix == '.parquet':
                table = pq.read_table(file)
                df = table.to_pandas()
                for idx in range(len(df)):
                    question = df.iloc[idx]['question']
                    question_audio = df.iloc[idx]['question_audio']
                    answer = df.iloc[idx]['answer']
                    samp = {
                        "conversations": [
                            {
                                "role": "user",
                                "content": question + '\n' + self.audio_token
                            },
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        ],
                        "audio": question_audio['bytes'] # not wave, is byte 
                    }
                    samples.append(samp)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _create_chat_prompt(self, conversations):
        return self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        audio_bytes = sample['audio']
        
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids
        origin_len = len(input_ids)
        input_ids = input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)
        
        padding_mask = [1] * origin_len + [0] * (self.max_length - origin_len)
        padding_mask = padding_mask[:self.max_length]

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        padding_mask = torch.tensor(padding_mask[1:], dtype=torch.long)
        
        audio_tensors = []
        # 1. turn the audio bytes to wave
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

        # if sample_rate is not 16000 for whisper
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # 2. turn the wave to Mel Spectrogram
        input_features = self.processor.feature_extractor(waveform[0], 
                                           padding='max_length',
                                           return_tensors='pt',
                                           sampling_rate=sample_rate,
                                           return_attention_mask=True) 
        # input_features represent audio input in transformers, just like pixel_values represent image
        # this is the step turning wave to mel, not audio input features
        feature_attention_mask = input_features.pop("attention_mask")
        
        return X, Y, loss_mask, feature_attention_mask, padding_mask, input_features['input_features']