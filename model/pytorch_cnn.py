# import package
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# create fully connected neural network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# create convolutional neural network
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )  # 8*28*28
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 8*14*14
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )  # 16*14*14
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (batch, 8, 28, 28)
        x = self.pool(x)  # (batch, 8, 14, 14)
        x = F.relu(self.conv2(x))  # (batch, 16, 14, 14)
        x = self.pool(x)  # (batch, 16, 7, 7)
        x = x.reshape(x.shape[0], -1)  # (batch, 16, 7, 7) -> (batch, 16*7*7)
        # x = x.view(x.size(0), -1) # same effect as above code
        x = self.fc1(x)
        return x


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
in_channels = 1
num_classes = 10  # 0 ~ 9
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# load data
train_dataset = datasets.MNIST(
    root="data/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root="data/", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# initialize model
model = CNN().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train_network():
    # train network
    for epoch in range(num_epochs):
        for batch_id, (data, targets) in enumerate(tqdm(train_loader)):
            # get data to cuda if possible
            data = data.to(device=device)  # torch.Size([64, 1, 28, 28])
            targets = targets.to(device=device)  # torch.Size([64])

            # forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)

            # backward propagation
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()


# test network (check accuracy on test set)
def check_accuracy(loader, model):
    if loader.dataset.train:
        print(f"Checking accuracy on training data")
    else:
        print(f"Checking accuracy on testing data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            _, predictions = outputs.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100: .2f}%"
        )

        model.train()

        acc = num_correct / num_samples
        return acc


def save_model(model):
    file_path = "mnist_cnn.pth"
    print(f"saving model to path: {file_path}")
    torch.save(model, file_path)


if __name__ == "__main__":
    train_network()
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
    save_model(model)
