import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

class VariationalDropout(nn.Module):
    def __init__(self, p):
        super(VariationalDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        
        binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
        mask = binomial.sample((x.shape[0], 1, x.shape[2])).to(x.device)
        #concatenate the mask on the second dimension
        mask_expanded = mask.expand_as(x)
        input_masked = x * mask_expanded * (1.0/(1-self.p))

        return input_masked

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, dropout_type="standard", weight_tying=False, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

        self.dropout_emb = nn.Dropout(emb_dropout) if dropout_type == "standard" else VariationalDropout(emb_dropout)
        self.dropout_out = nn.Dropout(out_dropout) if dropout_type == "standard" else VariationalDropout(out_dropout)

        self.pad_token = pad_index

        if weight_tying:
          self.output.weight = self.embedding.weight
        self.emb_dropout = emb_dropout
        self.out_dropout = out_dropout

    def forward(self, input_sequence):
        # input (batch_size, seq_len)
        emb = self.embedding(input_sequence)
        # emb (batch_size, seq_len, emb_size)
        # drop1 = self.dropout_1(emb)
        drop1 = self.dropout_emb(emb)
        lstm_out, _  = self.lstm(drop1)
        # lstm_out (batch_size, seq_len, hidden_size)
        # drop2 = self.dropout_2(lstm_out)
        drop2 = self.dropout_out(lstm_out)
        output = self.output(drop2).permute(0,2,1)
        # output (batch_size, vocab_len, seq_len)
        return output