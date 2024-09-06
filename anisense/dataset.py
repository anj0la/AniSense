import pandas as pd
from torch.utils.data import Dataset

class AnimeReviewDataset(Dataset):
    def __init__(self, annotations_file: str) -> None:
        self.reviews = pd.read_csv(annotations_file)

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int) -> tuple[list[int], int, int]:
        sequence = self.reviews.iloc[idx, 0]
        label = self.reviews.iloc[idx, 1]
        text_length = len(sequence)
        return sequence, label, text_length