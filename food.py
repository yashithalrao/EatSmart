import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# Load the pre-trained model and dataset
model = torchvision.models.resnet50(pretrained=True)
num_foods = 3  # replace with the actual number of food categories in your dataset
model.fc = nn.Linear(model.fc.in_features, num_foods)
model.load_state_dict(torch.load('io_mobilenet.h5'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

food_categories = ['apple_pie', 'baklava', 'baby_back_ribs']  # replace with your food categories list

st.title("Food Detection Application")

uploaded_image = st.file_uploader("Upload an image", type="jpg")

if uploaded_image is not None:
    # Process the uploaded image
    image = Image.open(uploaded_image)
    image = transform(image).unsqueeze(0)

    # Perform food detection
    output = model(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the index with the highest probability
    max_probability, max_index = torch.max(probabilities, 0)
    food_category = food_categories[max_index]

    # Load the calorie information from a file or database
    calories = load_calorie_information(food_category)

    st.write(f"Estimated food category: {food_category}")
    st.write(f"Estimated calories: {calories}")

    # Display the uploaded image
    plt.imshow(image[0].permute(1, 2, 0))
    st.pyplot()
