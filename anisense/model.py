"""
File: model.py

Author: Anjola Aina
Date Modified: September 5th, 2024

This module defines an LSTM-based model for sentiment analysis using PyTorch.

The model has the following structure, given the vocabulary size is 1000 with all default values:
    SentimentLSTM(
    (embedding): Embedding(1000, 64)
    (lstm): LSTM(64, 256, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
    (mlp): ModuleList(
        (0): Linear(in_features=256, out_features=128, bias=True)
        (1): Linear(in_features=128, out_features=64, bias=True)
        (2): Linear(in_features=64, out_features=2, bias=True)
    )
    (relu): ReLU()
    (dropout): Dropout(p=0.2, inplace=False)
    )
"""
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    An LSTM-based model for sentiment analysis.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimensionality of the embeddings. Default is 64.
        lstm_hidden_dim (int): The number of features in the hidden state of the LSTM. Default is 256.
        num_lstm_layers (int): The number of recurrent layers in the LSTM. Default is 2.
        hidden_dims (list[int]): The number of hidden states in the multi-layer perceptron. The first value should be equal to the lstm_hidden_dim if the LSTM is bidirectional, otherwise, halve it. Default is [256, 128, 64].
        output_dim (int): The size of the output layer. Default is 2.
        dropout (float): The dropout probability. Default is 0.2.
        batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature). Default is True.
        bidirectional (bool): If True, becomes a bidirectional LSTM. Default is True.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 64, lstm_hidden_dim: int = 256, num_lstm_layers: int = 2, hidden_dims: list[int] = [256, 128, 64], output_dim: int = 2, dropout: int = 0.2, batch_first: bool = True, bidirectional = True):
        super(SentimentLSTM, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Long-term short memory (LSTM) layer(s)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        
        # MLP layer
        self.mlp = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i + 1]) if i < len(hidden_dims) - 1 else nn.Linear(hidden_dims[i], output_dim) for i in range(len(hidden_dims))])
        
        # Activation function (for MLP layer)
        self.relu = nn.ReLU()
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
                        
    def forward(self, input, input_lengths):
        """
        Implements the forward pass for the SentimentLSTM model.

        Args:
            input (torch.Tensor): The input tensor containing the text data. The shape of the input should be the following: (batch size, sequence length).
            input_lengths (torch.Tensor): A tensor containing the lengths of each sequence in the batch.

        Returns:
            torch.Tensor: The output of the model after passing through the LSTM and MLP layers.
        """
        # Embedding layer
        embeddings = self.embedding(input) 
        
        # Temporarily moving lengths to CPU
        input_text_lengths_cpu = input_lengths.cpu()
        
        # Packed embeddings
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(input=embeddings, lengths=input_text_lengths_cpu, batch_first=True, enforce_sorted=False)
        
        # LSTM layer (we only care about the last hidden layer, not the packed output or cell)
        _, (hidden, _) = self.lstm(packed_embeddings)
        
        # Getting the last hidden state 
        if self.lstm.bidirectional:
            # Concatenating the last forward and backward hidden states
            hidden_lstm = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden_lstm = hidden[:,-1]
        
        # MLP layer
        output = hidden_lstm
        for fc in self.mlp:
            # Feed forward
            output = self.relu(fc(output))
            # Dropout layer
            output = self.dropout(output)
        
        return output
    
# Seeing the model layers
# print(SentimentLSTM(1000))
