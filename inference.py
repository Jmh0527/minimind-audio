import argparse
import os
import torch
import torchaudio
import random
import numpy as np
import warnings
from transformers import AutoTokenizer, TextStreamer, GenerationConfig
from model.model_audio import MiniMindALM, ALMConfig

warnings.filterwarnings("ignore")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_model(config: ALMConfig, device, args):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True, local_files_only=True)
    ckpt = f"{args.save_dir}/sft_alm.pth"

    model_config = ALMConfig.from_pretrained("Qwen/Qwen3-0.6B")
    model = MiniMindALM(model_config)
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    print(f"ALM参数量：{count_parameters(model) / 1e6:.3f} 百万")
    return model.eval().to(device), tokenizer, model.processor


def chat_with_alm(prompt, audio_path, model, tokenizer, processor, args):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        sample_rate = 16000

    input_features = processor.feature_extractor(
        waveform[0],
        sampling_rate=16000,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )
    feature_attention_mask = input_features["attention_mask"]
    input_features = input_features["input_features"]

    AUDIO_TOKEN = '$' * 750
    messages = [{"role": "user", "content": prompt + AUDIO_TOKEN}]
    chat_prompt = tokenizer.apply_chat_template(messages, 
                                                tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=False)

    inputs = tokenizer(chat_prompt, 
                       return_tensors="pt", 
                       truncation=True).to(args.device)
    input_features = input_features.to(args.device)
    feature_attention_mask = feature_attention_mask.to(args.device)

    print(f'[Audio]: {audio_path}')
    print('🤖️: ', end='')

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True
    )

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        input_features=input_features.unsqueeze(0), # 训练中的input_features第一维度是堆叠起来的
        attention_mask=inputs["attention_mask"],
        feature_attention_mask=feature_attention_mask,
        generation_config=generation_config,
        streamer=TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True) if args.stream else None
    )

    if not args.stream:
        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(response)
    print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with MiniMind-ALM")
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--max_new_tokens', default=128, type=int)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--audio_path', default='./example.wav', type=str)

    args = parser.parse_args()
    setup_seed(42)

    model_config = ALMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe
    )
    model, tokenizer, processor = init_model(model_config, args.device, args)

    question = "Answer the question according to the audio: \n"
    chat_with_alm(question, args.audio_path, model, tokenizer, processor, args)
