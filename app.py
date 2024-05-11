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


st.title("MNIST Handwritten Digit Recognition")
st.write("This is a simple web app to recognize handwritten digits using a simple model trained on the MNIST dataset.")
st.write("Please draw a digit between 0 and 9 and click on the 'Recognize' button to see the prediction.")


# Load the model
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN()
    checkpoint = torch.load("model/mnist_cnn.pth", map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


model = load_model()

# Add the canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",  # Fixed fill color with some opacity
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
    img = cv2.resize(img, (28, 28))  # img.shape: (28, 28, 4)

    # convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # img.shape: (28, 28)

    # show the image
    st.image(gray_img, width=28)

    # convert the image to a tensor -> # torch.Size([1, 1, 28, 28])
    tensor_img = torch.tensor(gray_img)  # torch.Size([28, 28])
    tensor_img = tensor_img.unsqueeze(0)  # torch.Size([1, 28, 28])
    tensor_img = tensor_img.unsqueeze(0).float()  # torch.Size([1, 1, 28, 28])

    # Make the prediction
    prediction = model(tensor_img)
    # print(f"Prediction Probabilities: {prediction}")

    # softmax
    prediction = F.softmax(prediction, dim=1)  # normalize the prediction
    print(f"Prediction Probabilities: {prediction}")

    # show the bar chart
    hist_values = prediction.detach().numpy().flatten().tolist()
    hist_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data = dict(zip(hist_labels, hist_values))
    st.bar_chart(data)

    prediction = torch.argmax(prediction, dim=1)  # return 1d tensor
    prediction = prediction.item()  # return the value as a number
    print(f"Prediction: {prediction}")
    st.write(f"Prediction: {prediction}")
