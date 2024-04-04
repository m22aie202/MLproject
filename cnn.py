import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, precision_recall_fscore_support
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_val_set = datasets.USPS(root='./data', train=True, download=True, transform=transform)
test_set = datasets.USPS(root='./data', train=False, download=True, transform=transform)

train_val_size = len(train_val_set)
train_size = int(0.8 * train_val_size)
val_size = train_val_size - train_size
train_set, val_set = random_split(train_val_set, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, config['conv_channels'], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(config['conv_channels'], config['conv_channels']*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['conv_channels']*2 * 4 * 4, config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

writer = SummaryWriter()

mlp_model = MLP().to(device)

criterion = nn.CrossEntropyLoss()
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

def train(model, optimizer, criterion, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        val_acc, val_precision, val_recall, val_conf_matrix = evaluate(model, val_loader)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Precision/val', val_precision, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())

    acc = correct / total
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    return acc, precision, recall, conf_matrix

cnn_configs = [
        {'conv_channels': 32, 'hidden_size': 64},  # Configuration 1
        {'conv_channels': 64, 'hidden_size': 128},  # Configuration 2
        {'conv_channels': 128, 'hidden_size': 256}   # Configuration 3
]


for i, cnn_config in enumerate(cnn_configs):
    cnn_model = CNN(cnn_config).to(device)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    train(cnn_model, cnn_optimizer, criterion, train_loader, val_loader)                            
    cnn_acc, cnn_precision, cnn_recall, cnn_conf_matrix = evaluate(cnn_model, test_loader)                                                        
    writer.add_scalar(f'Accuracy/CNN_Config_{i+1}', cnn_acc)
    fig_cnn = plot_confusion_matrix(cnn_conf_matrix, classes=range(10))
    writer.add_figure(f'Confusion matrix/CNN_Config_{i+1}_test', fig_cnn)


train(mlp_model, mlp_optimizer, criterion, train_loader, val_loader)

mlp_acc, mlp_precision, mlp_recall, mlp_conf_matrix = evaluate(mlp_model, test_loader)

writer.add_scalar('Accuracy/MLP', mlp_acc)
writer.add_scalar('Precision/MLP', mlp_precision)
writer.add_scalar('Recall/MLP', mlp_recall)

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt.gcf()

fig_mlp = plot_confusion_matrix(mlp_conf_matrix, classes=range(10))


writer.add_figure('Confusion matrix/MLP_test', fig_mlp)

writer.close()
