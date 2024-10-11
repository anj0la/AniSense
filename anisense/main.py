"""
File: main.py

Author: Anjola Aina
Date Modified: October 10th, 2024

This file contains all the necessary functions used to make a prediction on the model. It uses the saved weights, vectorizer, and label encoder obtained during
training.
"""
import joblib
import numpy as np
import pandas as pd
import torch 
from anisense.preprocess import clean_review
from anisense.model import SentimentLSTM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

def load_model(path: str, vectorizer: CountVectorizer) -> SentimentLSTM:
    """
    Loads the model.

    Args:
        path (str): The path to the model.
        vectorizer (CountVectorizer): The vectorizer used during training.

    Returns:
        SentimentLSTM: The loaded model.
    """
    model = SentimentLSTM(len(vectorizer.vocabulary_)) # Ensureing that the length of vocab of loaded model is the SAME as when training the model
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

def preprocess_sentence(sentence: str, vectorizer: CountVectorizer) -> np.ndarray:
    """
    Prepares the input by preprocessing it, and converting it into a textual format suitable for the model.

    Args:
        sentence (str): The raw sentence.
        vectorizer (CountVectorizer): The vectorizer used during training.

    Returns:
        np.ndarray: The encoded sentence, represented as a dense matrix.
    """
    data = {'review': [sentence]}
    df = pd.DataFrame(data)
    
    processed_data = clean_review(df)
    vectorized_text = vectorizer.transform(processed_data)

    return vectorized_text.toarray().squeeze() # Dense matrix

def predict(model: SentimentLSTM, sentence: str, vectorizer: CountVectorizer, le: LabelEncoder) -> str:
    """
    Predicts the sentiment of a sentence.

    Args:
        model (SentimentLSTM): The pre-trained model.
        sentence (str): The raw sentence.
        vectorizer (CountVectorizer): The vectorizer used during training.
        le (LabelEncoder): The label encoder used during training.

    Returns:
        str: The prediction. 
    """
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print('defined the device')
    
    # Preprocess the input sentence
    encoded_sentence = preprocess_sentence(sentence, vectorizer)
    tensor = torch.LongTensor(encoded_sentence).to(device)
    tensor = tensor.unsqueeze(1).T  # Reshaping in form of batch, number of words
    tensor_length = torch.LongTensor([len(encoded_sentence)]).to(device)  
    
    print('preprocessed input')

    # Set the model to evaluation mode
    model.eval()
    
    # Make the prediction, returning the logits
    with torch.no_grad(): # Disable gradient computation for inference
        logits = model(tensor, tensor_length)
        
    print('got logits')
    print(logits)
    print(torch.nn.functional.softmax(logits))
    
    # Get the predicted class index using argmax
    predicted_index = torch.argmax(logits, dim=1).item()  # Use item() to get the Python number
    print(predicted_index)
    print(le.classes_)
    
    # Inverse transform to get the original label
    predicted_label = le.inverse_transform([predicted_index])  # Wrap in a list
    
    print('made prediction')

    return predicted_label[0]  # Return the original label

def main() -> None:
    """
    The main entry point of the file.
    """
    # Sentence and paths
    sentence = 'I hate watching My Hero Academia, it was so boring.'
    path_to_model = 'model/model_saved_state.pt'
    path_to_vectorizer = 'model/vectorizer.pkl'
    path_to_encoder = 'model/label_encoder.pkl'
    
    # Load the saved vectorizer, label encoder and model
    vectorizer = joblib.load(path_to_vectorizer)
    le = joblib.load(path_to_encoder)
    model = load_model(path_to_model, vectorizer)
        
    # Get the prediction and print it
    prediction = predict(model, sentence, vectorizer, le)
    print(f'Sentence: {sentence} \nPrediction: {prediction}')

if __name__ == '__main__':
    main()