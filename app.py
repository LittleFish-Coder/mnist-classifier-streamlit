import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# define the CNN model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


st.title("MNIST Handwritten Digit Recognition")
st.write(
    "This is a simple web app to recognize handwritten digits using a simple model trained on the MNIST dataset."
)
st.write(
    "Please draw a digit between 0 and 9 and click on the 'Recognize' button to see the prediction."
)


# Load the model
def load_model():
    model = Model()
    model.load_state_dict(torch.load("mnist.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


model = load_model()

# Add the canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color="black",
    background_color="#fff",
    width=150,
    height=150,
    drawing_mode="freedraw",
    key="canvas",
)

# Add a button to get the prediction
if st.button("Recognize"):
    # Get the image data from the canvas
    img = canvas_result.image_data.astype(np.uint8)

    # resize the image to 28x28
    img = cv2.resize(img, (28, 28))  # (28, 28, 4)

    # convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (28, 28)

    # convert the image to a tensor
    gray_img = torch.tensor(gray_img).unsqueeze(0).float()  # torch.Size([1, 28, 28])

    # Make the prediction
    prediction = model(gray_img)
    prediction = torch.argmax(prediction, dim=1)  # return 1d tensor
    prediction = prediction.item()  # return the value as a number
    st.write(f"Prediction: {prediction}")
