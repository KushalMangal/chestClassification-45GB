import streamlit as st
from PIL import ImageOps, Image
import io
import base64
import numpy as np
from util1 import classify
import torch  # Add PyTorch import
import random

# Custom CSS with improved color scheme and borders
st.markdown(
    """
    <style>
        .stApp {
            background-color: #e0f2f1;  /* Light blue background */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }

        .stApp h1 {
            color: #2e8b57;  /* Forest green for headings */
            text-align: center;
            margin-bottom: 20px;
            font-family: 'Arial', sans-serif;
        }

        .stApp h2 {
            color: #3498db;  /* Bright blue for subheadings */
            text-align: center;
            font-weight: bold;
            font-size: 18px;
        }

        .stApp h3 {
            color: #9b59b6;  /* Soft purple for descriptions */
            text-align: center;
            font-style: italic;
            font-size: 16px;
        }

        .stApp header {
            background-color: #2e8b57;  /* Forest green for header */
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 16px;
        }

        .stApp .stFileUploader {
            background-color: #e0f2f1;  /* Light blue for file uploader */
            border: 2px solid #2e8b57;  /* Forest green border */
            border-radius: 5px;
            padding: 10px;
        }

        .stApp .stFileUploader label {
            color: #2e8b57;  /* Forest green for file uploader label */
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Centered header
st.markdown('<h1>NIH Chest X-Ray Classification</h1>', unsafe_allow_html=True)

# Description
st.markdown('<h2>The image classification model categorizes images into the following:</h2>', unsafe_allow_html=True)
classes_list =  ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"]
st.markdown('<h3>' +  ' • '.join(classes_list) + '</h3>', unsafe_allow_html=True)  # Join class names with • as separator

# Upload section with clear instructions
st.header('Please upload a chest X-ray image for classification (JPEG, JPG, or PNG only)')
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load the PyTorch model
model = torch.load("object_detection_chestxray_pred_model.pth", map_location=torch.device('cpu'))  # Load model on CPU
model.eval()  # Make sure the model is in evaluation mode

# Load class names
class_names = classes_list

# Display uploaded image
if file is not None:
    image = Image.open(file).convert('RGB')
    image = image.resize((248, 248))

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()

    # Encode the byte array to base64 string
    img_base64 = base64.b64encode(img_byte).decode()

    # Create three columns
    col1, col2, col3 = st.columns([1.2,2,1])

    # Use the middle column to display the image with a border
    with col2:
        # Custom HTML for displaying the image with a border
        st.markdown(f"""
        <div style="border:5px solid #A799B7; display: inline-block;">
            <img src="data:image/jpeg;base64,{img_base64}" alt="Beautiful Image" style="max-width: 100%;">
        </div>
        """, unsafe_allow_html=True)

    # Classify the image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification results
    st.write("## {}".format(class_name))
    st.write("### score: {}".format(conf_score))

# Function to generate response (you can replace this with your actual model call)
def generate_response(input_text):
    # Placeholder for response generation logic
    return "Response to: " + input_text

# Button to generate response
if st.button('Generate Report'):
    # Call the function to generate response
    response = generate_response("Your input text")

    # Display the response
    st.write(response)
