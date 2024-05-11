# import packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create Convolution Neural Network
class CNN(nn.Module):
    """
    The output shape(Height and Width) after Conv2d: [(W-K+2P)/S]+1d
    """

    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)  # [64, 1, 28, 28] -> [64, 8, 26, 26]
        x = F.relu(x)
        x = self.pool(x)  # [64, 8, 26, 26] -> [64, 8, 13, 13]
        x = self.conv2(x)  # [64, 8, 13, 13] -> [64, 16, 11, 11]
        x = F.relu(x)
        x = self.pool(x)  # [64, 16, 11, 11] -> [64, 16, 5, 5]
        x = x.reshape(x.shape[0], -1)  # [64, 16, 5, 5] -> [64, 400]
        x = self.fc1(x)  # [64, 400] -> [64, 10]
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

            # inference
            scores = model(data)  # [64, 10]

            _, predictions = scores.max(1)  # get the highest label for each input
            num_correct += (predictions == targets).sum()  # where the corresponding prediction is same to the ground truth
            num_samples += predictions.size(0)  # the current batch_szie

        accuracy = float(num_correct) / float(num_samples)
        print(f"Get {num_correct}/{num_samples} with accuracy {accuracy*100: .2f}%")

        model.train()
        return accuracy


# save model
def save_checkpoint(model, optimizer, filename="mnist_cnn.pth"):
    print("=> Saving checkpoint")
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, filename)
    print("=> Checkpoint saved")


if __name__ == "__main__":
    # debug test
    # model = CNN(28, 10)
    # x = torch.randn(64, 1, 28, 28)
    # output = model(x)
    # print(output)
    # print(output.shape)

    # hyperparameters
    in_channels = 1
    num_classes = 10
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 5

    # load dataset
    train_dataset = datasets.MNIST(root="../dataset/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="../dataset/", train=False, transform=transforms.ToTensor(), download=True)

    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # initialize network
    model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train network
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, targets) in enumerate(train_loader):
            # get data to cuda
            data = data.to(device)
            targets = targets.to(device)

            # print(data.shape)  # [batch_size, channel, height, width]: [64, 1, 28, 28]
            # print(target.shape)  # [batch_size]: [64] each value correspond to digital label from 0~9

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
    save_checkpoint(model, optimizer)  # save the model
