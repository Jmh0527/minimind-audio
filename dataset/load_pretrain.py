import torchaudio
from torch.utils.data import DataLoader

# 本地 LibriSpeech 数据集路径（只需到 LibriSpeech 文件夹）
root = "/home/kh31/transformers_design/minimind-audio/dataset"

# 加载 train-clean-100 子集，download=False 表示不自动下载
dataset = torchaudio.datasets.LIBRISPEECH(root=root, url="train-clean-100", download=False)

# 查看样本数量
print(f"Samples: {len(dataset)}")

# 查看一个样本的数据结构
waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = dataset[0]
print(f"Transcript: {transcript}")
print(f"Waveform shape: {waveform.shape}, Sample rate: {sample_rate}")

