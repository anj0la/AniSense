"""
File: preprocess.py

Author: Anjola Aina
Date Modified: September 6th, 2024

This file contains all the necessary functions used to preprocess the collected data. It includes functions to tokenize the data,
encode the text and labels, 
"""
import csv
import emoji
import json
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.lemmatize_text import lemmatize_text

def load_vocabulary(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
    return vocab

def save_to_csv(cleaned_text: list[str], labels: list[str], file_path: str) -> None:
    """
    Saves the text data and their labels into a CSV file.
        
    Args:
    cleaned text (list[str]: The cleaned text data.
    labels (list[str]): The labels for each piece of text.
    file_path (str): The path to save the CSV file.
    """
    fields = ['text', 'label']
    rows = []
    for sentence, label in zip(cleaned_text, labels):
        rows.append({'text': sentence, 'label': label})
    with open(file=file_path, mode='w', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        csv_file.close()  
            
def save_vocabulary(vocab, path):
    with open(path, 'w') as f:
        json.dump(vocab, f)        

def tokenize_data(cleaned_data: list[str]) -> tuple[list[list[str]], dict[str, int]]:
    """
    Tokenizes the text data into words and creates a vocabulary dictionary.
        
    Args:
        cleaned_data (list[str]): The preprocessed data.
        
    Returns:
        tuple: A tuple containing:
            - list[list[str]]: The tokenized data.
            - dict[str, int]: The vocabulary dictionary mapping tokens to indices.
    """
    tokenized_data = []
    for sentence in cleaned_data:
        tokenized_sentence = [word_tokenize(term) for term in sent_tokenize(sentence)]
        # filtered_sentence = [[word for word in sentence if word.strip()] for sentence in tokenized_sentence]
        tokenized_data.extend(tokenized_sentence)
            
    all_tokens = [token for sentence in tokenized_data for token in sentence]
    vocab = {token: idx for idx, token in enumerate(set(all_tokens))}
    # vocab['<UNK>'] = len(vocab) + 1
            
    return tokenized_data, vocab

def _encode_token(tokenized_sentence: list[str], vocab: dict) -> list[int]:
    """
    Encodes a token vector into a list of integers using the vocabulary dictionary.
        
    Args:
        token_vector (list[str]): A tokenized sentence.
        vocab (dict): The vocabulary dictionary.
        
    Returns:
        list[int]: The encoded token vector.
    """
    return [vocab.get(token) for token in tokenized_sentence]
        
def encode_tokens(tokenized_data: list[list[str]], vocab: dict) -> list[list[int]]:
    """
    Encodes multiple token vectors into lists of integers using the vocabulary dictionary.
        
    Args:
        token_vectors (list[list[str]]): The tokenized text data.
        vocab (dict): The vocabulary dictionary.
        
    Returns:
        list[list[int]]: The encoded token vectors.
    """
    return [_encode_token(tokenized_sentence, vocab) for tokenized_sentence in tokenized_data]

def encode_labels(labels: list[str]) -> list[int]:
    """
    Encodes string labels into their integer counterparts.
    
    The 'positive' label maps to 0, the 'negative' label maps to 1, and the 'neutral' label maps to 2.
        
    Args:
        labels (list[str]): The list of labels. 
        
    Returns:
        list[int]: The encoded list of labels.
    """
    encoded_labels = []
    for i in range(len(labels)):
        if labels[i] == 'positive':
            encoded_labels.append(0)
        elif labels[i] == 'negative':
            encoded_labels.append(1)
        else: # labels[i] == 'neutral
            encoded_labels.append(2)
    return encoded_labels

def prepare_input(tokenized_data, labels, vocab):
    # Encode tokenized data
    encoded_sequences = encode_tokens(tokenized_data, vocab)
    
    # Encode labels (positive = 0, negative = 1, neutral = 2)
    encoded_labels = encode_labels(labels)
    
    return encoded_sequences, encoded_labels

def get_labels(df: pd.DataFrame) -> list[str]:
    """
    Converts scalar scores of each review into a 'positive', 'negative' or 'neutral' label.

    Args:
        df (pd.DataFrame): The reviews Pandas DataFrame.

    Returns:
        list[str]: The list of labels.
    """
    scores = df['score']
    labels = []
    
    for score in scores:
        if score >= 7: # Scores ranging from 7 to 10 are positive
            labels.append('positive')
        elif score < 7 and score >= 5: # Scores ranging from 5 to 6 are neutral
            labels.append('neutral')
        else: # Scores ranging from 1 to 4 are negative
            labels.append('negative')
            
    return labels
            
def preprocess(file_path: str) -> list[str]:
    """
    Preprocesses the text data, returning the processed data.
     
    This function preprocesses the text data by converting the text to lowercase and emojis to text, removing punctuation, special characters,
    links, email addresses and applying lemmatization.
        
    Args:
        file_path (str): The file path containing the text data.
        
    Returns:
        list[str]: The cleaned data.
    """
    df = pd.read_csv(file_path)
    data = df['text']
        
    # Convert the text to lowercase
    data = data.str.lower()
    
    # Remove punctuation and special characters
    data = data.replace(r'[.,;:!\?"\'`]', '', regex=True)
    data = data.replace(r'[@#\$%^&*\(\)\\/\+\-_=\\[\]\{\}<>]', '', regex=True)
    data = data.replace(r'[‘’“”]', '', regex=True)
    
    # Convert emojis to text
    data = data.apply(lambda x: emoji.demojize(x))
    data = data.replace(r':', '', regex=True)
        
    # Remove links and email addresses
    data = data.replace(r'http\S+|www\.\S+', '', regex=True)
    data = data.replace(r'\w+@\w+\.com', '', regex=True)
        
    # Remove stop words and apply lemmatization
    stop_words = set(stopwords.words('english'))
    data = data.apply(lambda sentence: ' '.join(word for word in sentence.split() if word not in stop_words))
    data = data.apply(lambda sentence: lemmatize_text(sentence))
    
    # Extract labels
    labels = get_labels(df)
            
    return data.values, labels
    
# Running the code
""" preprocessed_data, labels = preprocess('data/reviews.csv')
tokenized_data, vocab = tokenize_data(cleaned_data=preprocessed_data)

#print(f'Data: {preprocessed_data[:5]}, Labels: {labels[:5]}')

#save_to_csv(preprocessed_data, labels, 'data/cleaned_reviews.csv')
#save_vocabulary(vocab=vocab, path='data/vocab.json')

#vocab = load_vocabulary(path='data/vocab.json')
#df = pd.read_csv('data/cleaned_reviews.csv')
#tokenized_data = df['text'].values
#labels = df['label']

encoded_sequences, encoded_labels = prepare_input(tokenized_data, labels, vocab)

print('\n\n\n################################ TESTING ENCODING FUNCTION ################################\n\n\n')
print('Tokenized data: ', tokenized_data[:1])
print('Encoded data: ', encoded_sequences[:1])
print('Encoded labels: ', encoded_labels[:1])

print('Testing: ', vocab.get('story')) """