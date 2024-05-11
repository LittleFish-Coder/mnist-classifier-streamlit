# import packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# ceate Fully Neural Network
class NN(nn.Module):
    def __init__(self, input_size=784, num_classes=10):  # input_size: 28*28=784, num_classes: 0~9
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.softmax(x)
        return x


# set device
device = "cuda" if torch.cuda.is_available() else "cpu"


# check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.to(device)
            # reshape
            data = data.reshape(data.shape[0], -1)
            # inference
            scores = model(data)  # [64, 10]

            _, predictions = scores.max(1)  # get the highest label for each input
            num_correct += (predictions == targets).sum()  # where the corresponding prediction is same to the ground truth
            num_samples += predictions.size(0)  # the current batch_szie

        accuracy = float(num_correct) / float(num_samples)
        print(f"Get {num_correct}/{num_samples} with accuracy {accuracy*100: .2f}%")

        model.train()
        return accuracy


if __name__ == "__main__":
    # debug test
    model = NN(784, 10)
    x = torch.randn(64, 784)
    output = model(x)
    print(output)
    print(output.shape)

    # hyperparameters
    input_size = 28 * 28
    num_classes = 10
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 1

    # load dataset
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # initialize network
    model = NN(input_size=input_size, num_classes=num_classes).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # get data to cuda
            data = data.to(device)
            targets = targets.to(device)

            # print(data.shape)  # [batch_size, channel, height, width]: [64, 1, 28, 28]
            # print(target.shape)  # [batch_size]: [64] each value correspond to digital label from 0~9

            # reshape the data to match model input
            data = data.reshape(data.shape[0], -1)

            # forward pass
            scores = model(data)  # get the predicted label
            loss = criterion(scores, targets)  # calculate the loss between prediction and ground truth

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            # gradient descent
            optimizer.step()

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
