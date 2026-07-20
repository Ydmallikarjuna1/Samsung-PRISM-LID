from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class VoxLinguaDataset(Dataset):
    """
    VoxLingua107 Dataset

    Returns:
        audio_path : str
        label      : int
    """

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        audio_path = row["path"]
        label = int(row["label"])

        return audio_path, label