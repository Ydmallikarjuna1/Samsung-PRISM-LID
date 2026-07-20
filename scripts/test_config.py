from configs.config import AudioConfig, ModelConfig, TrainConfig

print(AudioConfig.sample_rate)
print(AudioConfig.num_mel_bins)

print(ModelConfig.channels)
print(ModelConfig.embedding_dim)

print(TrainConfig.batch_size)
print(TrainConfig.learning_rate)