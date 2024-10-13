"""
File: model.py

Author: Anjola Aina
Date Modified: September 6th, 2024

This module defines an LSTM-based model for sentiment analysis using PyTorch.

The model has the following structure, given the vocabulary size is 1000 with all default values:
    SentimentLSTM(
    (embedding): Embedding(1000, 64)
    (lstm): LSTM(64, 256, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
    (mlp): ModuleList(
        (0): Linear(in_features=256, out_features=128, bias=True)
        (1): Linear(in_features=128, out_features=64, bias=True)
        (2): Linear(in_features=64, out_features=3, bias=True)
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
        output_dim (int): The size of the output layer. Default is 3.
        dropout (float): The dropout probability. Default is 0.2.
        batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature). Default is True.
        bidirectional (bool): If True, becomes a bidirectional LSTM. Default is True.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 256, lstm_hidden_dim: int = 64, num_lstm_layers: int = 1, hidden_dims: list[int] = [64, 32], output_dim: int = 3, dropout: float = 0.0, batch_first: bool = True, bidirectional = True):
        super(SentimentLSTM, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Long-term short memory (LSTM) layer(s)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        
        # MLP layer
        if self.lstm.bidirectional:
            self.mlp = nn.ModuleList(
            [nn.Linear(lstm_hidden_dim * 2, hidden_dims[0])] # Accounting for the concatenation of forward and backward hidden states 
            ).extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]) if i < len(hidden_dims) - 1 else nn.Linear(hidden_dims[i], output_dim) for i in range(len(hidden_dims))])
        else:
            self.mlp = nn.ModuleList(
            [nn.Linear(lstm_hidden_dim, hidden_dims[0])] # First layer is lstm hidden 
            ).extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]) if i < len(hidden_dims) - 1 else nn.Linear(hidden_dims[i], output_dim) for i in range(len(hidden_dims))])
        
        # Final layer 
        # self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)
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
        
        # Print the input shape and embedding output shape
        # print("Input tensor shape (batch size, sequence length):", input.shape)
        # print("Embedding output shape:", embeddings.shape)
        
        # Packed embeddings
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(input=embeddings, lengths=input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Print the shape of packed embeddings
        # print("Packed embeddings shape:", packed_embeddings.data.shape)
        
        # LSTM layer (we only care about the last hidden layer, not the packed output or cell)
        _, (hidden, _) = self.lstm(packed_embeddings)
        
        # Getting the last hidden state 
        if self.lstm.bidirectional:
            # Concatenating the last forward and backward hidden states
            hidden_lstm = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden_lstm = hidden[:,-1]
        
        
        # Print the shape of the last hidden state
        # print("Hidden LSTM state shape:", hidden_lstm.shape)
        
        # mlp layer (output is one neuron)
        #output = self.fc(hidden_lstm)
        
        # return output
        
        # MLP layer
        output = hidden_lstm
        
        for fc in self.mlp:
            # Feed forward
            output = self.relu(fc(output))
            # Dropout layer
            # output = self.dropout(output)
            
        # print('final output shape: ', output.shape)
        
        return output
    