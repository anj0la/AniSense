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

def save_to_csv(cleaned_text: list[str], labels: list[str], file_path: str) -> None:
    """
    Saves the cleaned text and corresponding labels into a CSV file.
        
    Args:
    cleaned_text (list[str]: The cleaned text data.
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
            
def preprocess(file_path: str, output_file_path: str) -> None:
    """
    Preprocesses the text data, returning the processed data and corresponding labels.
     
    This function preprocesses the text data by converting the text to lowercase and emojis to text, removing punctuation, special characters,
    links, email addresses and applying lemmatization.
        
    Args:
        file_path (str): The file path containing the text data.
        output_file_path (str): The file path to put the cleaned text data into.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    if 'text' not in df.columns:
        raise ValueError('Expected column "text" in input file.')
    
    data = df['text']
        
    # Convert the text to lowercase
    data = data.str.lower()
    
    # Remove Unicode characters (non-ASCII)
    data = data.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
    
    # Remove punctuation, special characters, emails and links
    data = data.replace(r'[^\w\s]', '', regex=True)  # Removes non-alphanumeric characters except whitespace
    data = data.replace(r'http\S+|www\.\S+', '', regex=True)  # Remove URLs
    data = data.replace(r'\w+@\w+\.com', '', regex=True)  # Remove emails
    
    # Convert emojis to text
    data = data.apply(lambda x: emoji.demojize(x))
    data = data.replace(r':(.*?):', '', regex=True)
        
    # Remove stop words and apply lemmatization
    stop_words = set(stopwords.words('english'))
    data = data.apply(lambda sentence: ' '.join(lemmatize_text(word) for word in sentence.split() if word not in stop_words))

    # Extract labels
    labels = get_labels(df)
    
    # Save data to new CSV file
    save_to_csv(data.values, labels, output_file_path)
    
# Running the code
# preprocess(file_path='data/reviews.csv', output_file_path='data/cleaned_reviews.csv')
# save_to_csv(preprocessed_data, labels, 'data/cleaned_reviews.csv')

""" preprocessed_data, labels = preprocess('data/reviews.csv')
tokenized_data, vocab = tokenize_data(cleaned_data=preprocessed_data)

#print(f'Data: {preprocessed_data[:5]}, Labels: {labels[:5]}')

save_to_csv(preprocessed_data, labels, 'data/cleaned_reviews.csv')
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