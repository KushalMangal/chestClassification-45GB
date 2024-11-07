import streamlit as st
from PIL import ImageOps, Image
import io
import base64
from keras.models import load_model
import numpy as np
import torch

def classify(image, model, class_names):
    # Prepare the image
    image = ImageOps.fit(image, (248, 248), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = torch.tensor(normalized_image_array).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        prediction = model(data)

    print("Prediction output:", prediction)

    # Check if the prediction is a list and get the first element
    if isinstance(prediction, list) and len(prediction) > 0:
        prediction = prediction[0]  # Get the first element which is the dictionary
        
        if 'scores' in prediction and 'labels' in prediction:
            # Get the highest score and its index
            scores = prediction['scores']
            labels = prediction['labels']
            if len(scores) > 0:
                max_score_index = scores.argmax().item()
                class_name = class_names[labels[max_score_index].item()]  # Use labels to get class name
                confidence_score = scores[max_score_index].item()
                return class_name, confidence_score
            else:
                raise ValueError("No detections found in the prediction.")
        else:
            raise ValueError("Expected keys 'scores' and 'labels' in prediction dictionary.")
    else:
        raise ValueError("Unexpected prediction output format.")
