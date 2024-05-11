import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pytorch_cnn import CNN

# Set page config
st.set_page_config(
    page_title="MNIST Classifier",
    page_icon="🔢",
    layout="wide",
)


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
st.write("This is a simple web app to recognize handwritten digits using a simple model trained on the MNIST dataset.")
st.write("Please draw a digit between 0 and 9 and click on the 'Recognize' button to see the prediction.")


# Load the model
def load_model():
    model = torch.load("model/mnist_cnn.pth", map_location=torch.device("cpu"))
    model.eval()
    return model


model = load_model()

# Add the canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # Fixed fill color with some opacity
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
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

    # show the image
    st.image(gray_img, width=150)

    # convert the image to a tensor -> # torch.Size([1, 1, 28, 28])
    gray_img = torch.tensor(gray_img).unsqueeze(0).unsqueeze(0).float()

    # Make the prediction
    prediction = model(gray_img)
    print(prediction)

    # show the bar chart
    hist_values = F.softmax(prediction).detach().numpy().flatten().tolist()
    hist_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data = dict(zip(hist_labels, hist_values))
    st.bar_chart(data)

    prediction = torch.argmax(prediction, dim=1)  # return 1d tensor
    prediction = prediction.item()  # return the value as a number
    print(prediction)

    st.write(f"Prediction: {prediction}")
