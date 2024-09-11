import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class AnimeReviewDataset(Dataset):
    def __init__(self, annotations_file: str) -> None:
        self.reviews = pd.read_csv(annotations_file)
        self.vectorizer = CountVectorizer()
        self.vectorized_text = self.vectorizer.fit_transform(self.reviews['text'])
        self.le = LabelEncoder()
        self.encoded_labels = self.le.fit_transform(self.reviews['label'])
        self.vocabulary = self.vectorizer.vocabulary_


    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, int]:
        """
        Gets an item from the reviews.
        
        Note about squeezing the vectorized_text:
        After vectorizing the data, the shape of the text becomes (1, X), where 1 denotes the number of indices to index the vectorized text (initially indexed by single index) and X denotes the number of elements in the vectorized text.
        This means the vectorized text is indexed by two indicies (also known as a two dimensional array).
        
        We squeeze the array to remove the single-dimensional entries from the shape of the vectorized text. This gives us the text with X elements.
     
        Args:
            idx (int): The index of the item to retrieve the vectorized text and corresponding label.

        Returns:
            tuple[list[int], int, int]: _description_
        """
        sequence = self.vectorized_text[idx].toarray().squeeze() # Dense matrix
        label = self.encoded_labels[idx]
        sequence_length = len(sequence)
        return sequence, label, sequence_length
    

""" labels = ['positive', 'negative', 'neutral']
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
print(type(encoded_labels))
# print(le.classes_)
actual_labels = le.inverse_transform(encoded_labels)
print(actual_labels)

dataset = AnimeReviewDataset('data/test.csv')
# print(dataset.vectorized_text)
print(dataset.vectorized_text[0])

print(dataset.vectorized_text[0].toarray()) 

print(dataset.vectorized_text[0].toarray().squeeze()) 

print(dataset.vectorized_text[0].toarray().shape)
print(dataset.vectorized_text[0].toarray().squeeze().shape)

print(type(dataset.vectorized_text[0].toarray().squeeze()))
print(type(dataset.vectorized_text[0].toarray().squeeze()[0])) """