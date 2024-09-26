import torch
import torch.nn as nn

class MultiFeatureRNNModel(nn.Module):
    def __init__(self, feature_sizes, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(MultiFeatureRNNModel, self).__init__()
        self.branches = nn.ModuleList([
            nn.RNN(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers,
                   batch_first=True, dropout=dropout)
            for feature_size in feature_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * len(feature_sizes), output_size)

    def forward(self, x):
        # x: lista tensorów, każdy o kształcie (batch_size, seq_length, feature_size)
        outputs = []
        for i, branch in enumerate(self.branches):
            out, _ = branch(x[i])
            out = self.dropout(out)
            out = out[:, -1, :]  # Pobieramy wyjście z ostatniego kroku czasowego
            outputs.append(out)
        concatenated = torch.cat(outputs, dim=1)
        out = self.fc(concatenated)
        return out

class MultiFeatureGRUModel(nn.Module):
    def __init__(self, feature_sizes, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(MultiFeatureGRUModel, self).__init__()
        self.branches = nn.ModuleList([
            nn.GRU(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers,
                   batch_first=True, dropout=dropout)
            for feature_size in feature_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * len(feature_sizes), output_size)

    def forward(self, x):
        outputs = []
        for i, branch in enumerate(self.branches):
            out, _ = branch(x[i])
            out = self.dropout(out)
            out = out[:, -1, :]
            outputs.append(out)
        concatenated = torch.cat(outputs, dim=1)
        out = self.fc(concatenated)
        return out

class MultiFeatureLSTMModel(nn.Module):
    def __init__(self, feature_sizes, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(MultiFeatureLSTMModel, self).__init__()
        self.branches = nn.ModuleList([
            nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers,
                    batch_first=True, dropout=dropout)
            for feature_size in feature_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * len(feature_sizes), output_size)

    def forward(self, x):
        outputs = []
        for i, branch in enumerate(self.branches):
            out, _ = branch(x[i])
            out = self.dropout(out)
            out = out[:, -1, :]
            outputs.append(out)
        concatenated = torch.cat(outputs, dim=1)
        out = self.fc(concatenated)
        return out
