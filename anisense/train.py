"""
File: train.py

Author: Anjola Aina
Date Modified: September 6th, 2024

This file contains all the necessary functions used to train the model.
Only run this file if you want to add more training examples to improve the performance of the model.
Otherwise, use the pretrained model in the 'models' folder, called model_saved_weights.pt.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from anisense.model import SentimentLSTM
from anisense.dataset import AnimeReviewDataset
from anisense.preprocess import preprocess

def accuracy_score(y_true, y_pred):
    classes = torch.argmax(y_pred, dim=1)
    return torch.mean((classes == y_true).float())

def collate_batch(batch: tuple[list[int], int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of data for the DataLoader.

    This function takes a batch of sequences, labels, and lengths, converts them to tensors, 
    and pads the sequences to ensure they are of equal length. This is useful for feeding data 
    into models that require fixed-length inputs, such as LSTM models.

    Args:
        batch (list of tuples): A list where each element is a tuple containing three elements:
            - sequences (list of int): The sequence of token ids representing a piece of text.
            - labels (int): The label corresponding to the sequence.
            - lengths (int): The original length of the sequence.

    Returns:
        tuple: A tuple containing three elements:
            - padded_sequences (torch.Tensor): A tensor of shape (batch_size, max_sequence_length) 
              containing the padded sequences.
            - labels (torch.Tensor): A tensor of shape (batch_size,) containing the labels.
            - lengths (torch.Tensor): A tensor of shape (batch_size,) containing the original lengths 
              of the sequences.
    """
    encoded_sequences, encoded_labels, lengths = zip(*batch)
        
    # Converting the sequences, labels and sequence length to Tensors
    encoded_sequences = [torch.tensor(seq, dtype=torch.int64) for seq in encoded_sequences]
    encoded_labels = torch.tensor(encoded_labels, dtype=torch.float)
    lengths = torch.tensor(lengths, dtype=torch.long)
        
    # Padding sequences
    padded_encoded_sequences = nn.utils.rnn.pad_sequence(encoded_sequences, batch_first=True, padding_value=0.0)
    
    return padded_encoded_sequences, encoded_labels, lengths

def create_dataloaders(file_path: str, batch_size: int = 64, train_split: float = 0.8) -> tuple[DataLoader, DataLoader]:
    """
    Creates custom datasets and dataloaders for training and testing.

    Args:
        file_path (str): The path to the processed CSV file containing the data.
        batch_size (int): The size of the batches for the dataloaders. Default is 64.
        train_split (float): The proportion of the data to use for training. Default is 0.8.

    Returns:
        tuple: A tuple containing:
            - DataLoader: The dataloader for the training dataset.
            - DataLoader: The dataloader for the testing dataset.
    """
    # Create the custom dataset
    dataset = AnimeReviewDataset(file_path)
    
    # Split the dataset into training and testing sets
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create dataloaders for the training and testing sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    
    return train_dataloader, test_dataloader, dataset

def train_one_epoch(model: SentimentLSTM, iterator: DataLoader, optimizer: optim.SGD, device: torch.device) -> tuple[float, float]:
    """
    Trains the model for one epoch.

    Args:
        model (LSTM): The model to be trained.
        iterator (DataLoader): The DataLoader containing the training data.
        optimizer (optim.SGD): The optimizer used for updating model parameters.

    Returns:
        tuple: A tuple containing:
            - float: The average loss over the epoch.
            - float: The average accuracy over the epoch.
    """
    # Initialize the epoch loss and accuracy for every epoch 
    epoch_loss = 0
    epoch_accuracy = 0
    
    # Set the model in the training phase
    model.train()  
    
    # Go through each batch in the training iterator
    for batch in iterator:
        
        # Get the padded sequences, labels and lengths from batch 
        padded_sequences, labels, lengths = batch
        labels = labels.type(torch.LongTensor) # Casting to long
        
        # Move input and expected label to GPU
        padded_sequences = padded_sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        # Reset the gradients after every batch
        optimizer.zero_grad()   
                
        # Get expected predictions
        predictions = model(padded_sequences, lengths).squeeze()
    
        # print('labels: ', labels)
        print('predictions: ', predictions)        
                
        # Compute the loss
        loss = F.cross_entropy(predictions, labels)        
        
        # Compute metrics 
        accuracy = accuracy_score(y_true=labels.cpu().detach(), y_pred=predictions.cpu().detach()) 
        
        # Backpropagate the loss and compute the gradients
        loss.backward()       
        
        # Update the weights
        optimizer.step()      
        
        # Increment the loss and accuracy
        epoch_loss += loss.item()  
        epoch_accuracy += accuracy  
        
    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)

def evaluate_one_epoch(model: SentimentLSTM, iterator: DataLoader, device: torch.device) -> tuple[float, float]:
    """
    Evaluates the model on the validation/test set.

    Args:
        model (LSTM): The model to be evaluated.
        iterator (DataLoader): The DataLoader containing the validation/test data.
        device (torch.device): The device to train the model on.

    Returns:
        tuple: A tuple containing:
            - float: The average loss over the validation/test set.
            - float: The average accuracy over the validation/test set.
    """
    # Initialize the epoch loss and accuracy for every epoch 
    epoch_loss = 0
    epoch_accuracy = 0

    # Deactivate droput layers and autograd
    model.eval()
    with torch.no_grad():
        
        for batch in iterator:
            
            # Get the padded sequences, labels and lengths from batch 
            padded_sequences, labels, lengths = batch
            labels = labels.type(torch.LongTensor) # Casting to long
                        
            # Move sequences, expected labels and lengths to GPU
            padded_sequences = padded_sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
             # Get expected predictions
            predictions = model(padded_sequences, lengths).squeeze()
            
            # Compute the loss
            loss = F.cross_entropy(predictions, labels)        
            
            # Compute metrics 
            accuracy = accuracy_score(y_true=labels.cpu(), y_pred=predictions.cpu())
            
            # Keep track of metrics
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            
    
    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)
        
        
def run_gradient_descent(model: SentimentLSTM, train_iterator: DataLoader, test_iterator: DataLoader, device: torch.device, n_epochs: int = 10, 
               lr: float = 0.01, weight_decay: float = 0.0, model_save_path: str = 'model/saved_model.pt') -> None:
    """
    Train the model for multiple epochs and evaluate on the validation set.

    Args:
        model (SentimentLSTM): The model to be trained.
        train_iterator (DataLoader): The DataLoader containing the training data.
        valid_iterator (DataLoader): The DataLoader containing the validation data.
        n_epochs (int, optional): The number of epochs to train the model. Defaults to 5.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        weight_decay (float, optional): The weight decay for regularization. Defaults to 0.00.
        model_save_path (str, optional): The path to save the best model's weights. Defaults to 'model/saved_model.pt'.
    """
    best_test_loss = float('inf')
    optimizer = optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        
        # Train the model
        train_loss, train_accurary = train_one_epoch(model, train_iterator, optimizer, device)
        
        # Evaluate the model
        test_loss, test_accurary = evaluate_one_epoch(model, test_iterator, device)
        
        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # torch.save(obj=model, f=model_save_path)
        
        # Print train / test metrics
        print(f'\t Epoch: {epoch + 1} out of {n_epochs}')
        print(f'\t Train Loss: {train_loss:.3f} | Train Acc: {train_accurary * 100:.2f}%')
        print(f'\t Valid Loss: {test_loss:.3f} | Valid Acc: {test_accurary * 100:.2f}%')
        
def train(input_file_path: str, cleaned_file_path: str, train_ratio: int = 0.8, batch_size: int = 64) -> None:
    """
    Trains a LSTM model used for sentiment analysis.

    Args:
        file_path (str): The path to the cleaned reviews.
        train_split (int, optional): The proportion of the dataset to include in the train split. Defaults to 0.8.
        batch_size (int, optional): The batch size for each batch. Defaults to 64.
    """
    # Preprocess the file (if not already preprocessed)
    if not os.path.exists(cleaned_file_path):
        preprocess(file_path=input_file_path, output_file_path=cleaned_file_path)
    
    # Create the custom dataset
    dataset = AnimeReviewDataset(cleaned_file_path)
    
    # Split the dataset into training and testing sets
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create dataloaders for the training and testing sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    # Get the GPU device (if it exists)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Create the model
    model = SentimentLSTM(vocab_size=len(dataset.vocabulary)).to(device)
    print(model)

    # Run Gradient Descent
    run_gradient_descent(model=model, train_iterator=train_dataloader, test_iterator=test_dataloader, device=device) 

##### Running the code #####

train(input_file_path='data/reviews.csv', cleaned_file_path='data/cleaned_reviews.csv')
    
""" print('\n\nRunning train.py!\n\n')
# Get the training and testing dataloaders
train_dataloader, test_dataloader, dataset = create_dataloaders(file_path='data/test.csv')
vocab_len = len(dataset.vocabulary)

# Get the GPU device (if it exists)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = SentimentLSTM(vocab_size=vocab_len).to(device)

print(model)

# Run Gradient Descent
run_gradient_descent(model=model, train_iterator=train_dataloader, test_iterator=test_dataloader, device=device) """