import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=1000, shuffle=False)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = NN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for data, targets in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(data), targets)
        loss.backward()
        optimizer.step()
        
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            
    print(f"Epoch {epoch+1} | Accuracy: {100. * correct / len(test_loader.dataset):.2f}%")
