from dataclasses import dataclass


@dataclass
class AudioConfig:
    sample_rate = 16000
    num_mel_bins = 80
    frame_length = 25
    frame_shift = 10


@dataclass
class ModelConfig:
    channels = 512
    embedding_dim = 192
    scale = 8


@dataclass
class TrainConfig:
    batch_size = 32
    epochs = 50
    learning_rate = 1e-3
    weight_decay = 2e-5