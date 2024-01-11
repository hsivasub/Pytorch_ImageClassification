import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import boto3
import json
import timm
import cv2
import numpy as np


s3 = boto3.client('s3')

# Setup device agnostic code
#device = "cuda" if torch.cuda.is_available() else "cpu"
device='cpu'

loaded_model=torch.load('EffNetB0_10_99_90_LRS.pt')

def lambda_handler(event,context):
    Bucket = event['Bucket']
    Object = event['Object']
    ID= int(Object.split('/')[-1][:-4])

    response = s3.get_object(Bucket=Bucket, Key=Object)
    file_stream = response['Body'].read()
    image = cv2.imdecode(np.frombuffer(file_stream, np.uint8), -1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    # response = s3.get_object(Bucket=Bucket, Key=Object)
    # file_stream = response['Body']
    loaded_model.eval()  # Set the model to evaluation mode

    # Define a transformation for the input image (adjust as needed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class_names=['BLACK', 'BLUE', 'BROWN', 'GOLD', 'GRAY', 'GREEN', 'MULTI', 'RED', 'SILVER', 'TWOWHEELER', 'WHITE']

    # Move the input tensor to the same device as the model
    image = image.to(device)

    # Make a prediction
    with torch.no_grad():
        output = loaded_model(image)

    # Assuming it's a classification model, get the predicted class index
    predicted_class_index = torch.argmax(output).item()

    target_image_pred_probs = round(torch.softmax(output, dim=1).max().item(), 3)

    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]

    if target_image_pred_probs>=0.95:
        conf='High'
    else:
        conf='Low'

    # Color decoding

    if predicted_class_name in ['BLACK','GRAY']:
        final_color='BLACK/GRAY'
    elif predicted_class_name in ['RED','ORANGE']:
        final_color='RED/ORANGE'
    elif predicted_class_name in ['GOLD']:
        final_color='GOLD/YELLOW'
    else:
        final_color=predicted_class_name

    predict = {'ID':ID,
               'Predicted class': final_color,
               'ConfidenceScore': target_image_pred_probs,
               'Confidence':conf}

    save_to_s3 = s3.put_object(Key=f"VehicleColor/{ID}.json",
                               Bucket="write-result-bucket-test",
                               Body=(json.dumps(predict).encode('UTF-8'))
                               )

    return predict


# print(predict)
