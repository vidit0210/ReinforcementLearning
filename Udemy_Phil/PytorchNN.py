import torch.nn as nn  # for Layers
import torch.nn.functional as F  # For Activation Function
import torch.optim as optim  # for Optimizers
import torch as T


class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super().__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2)

        return layer3

    def learn(self, data, labels):
        self.optimizer.zero_grad()
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.device)

        predictions = self.forward(data)
        cost = self.loss(predictions, labels)
        cost.backward()
        self.optimizer.step()
