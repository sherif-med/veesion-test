from torch import nn
import torch

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # batch_first=True means input is (batch_size, sequence_length, features)

        self.fc = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # Initialize hidden and cell states
        # h0 shape: (num_layers, batch_size, hidden_size)
        # c0 shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass through LSTM
        # out shape: (batch_size, sequence_length, hidden_size)
        # (hn, cn) are the hidden and cell states for the last time step
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # We take the output from the last time step for classification
        # hn[-1, :, :] is the hidden state of the last layer at the last time step
        out = self.dropout(hn[-1, :, :])

        # Pass through fully connected layer
        logits = self.fc(out)
        return logits