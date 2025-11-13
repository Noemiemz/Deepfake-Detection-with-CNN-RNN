import torch
import torch.nn as nn
from torchvision import models

class DeepFakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove final FC layer
        self.rnn = nn.LSTM(input_size=2048, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, C, H, W)
        batch_size, seq_len = x.size(0), x.size(1)
        cnn_features = []
        for i in range(seq_len):
            features = self.cnn(x[:, i, :, :, :])  # (batch_size, 2048)
            cnn_features.append(features)
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, seq_len, 2048)
        _, (hidden, _) = self.rnn(cnn_features)
        output = self.fc(hidden[-1])
        return torch.sigmoid(output)
