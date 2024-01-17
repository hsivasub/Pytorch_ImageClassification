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
loaded_model.to(device)

def lambda_handler(event,context):
    Bucket = event['Bucket']
    Object = event['Object']
    ID= int(Object.split('/')[-1][:-4])

    response = s3.get_object(Bucket=Bucket, Key=Object)
    file_stream = response['Body'].read()
    image = cv2.imdecode(np.frombuffer(file_stream, np.uint8), -1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
    input_tensor = image_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    class_names=['a','b']


    # Make a prediction
    with torch.no_grad():
        output = loaded_model(input_tensor)

    # Assuming it's a classification model, get the predicted class index
    predicted_class_index = torch.argmax(output).item()

    target_image_pred_probs = round(torch.softmax(output, dim=1).max().item(), 3)

    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]

    

    predict = {'ID':ID,
               'Predicted class': predicted_class_name,
               'ConfidenceScore': target_image_pred_probs}

    return predict

