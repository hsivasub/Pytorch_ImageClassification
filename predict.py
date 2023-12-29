import os
import torch
from torchvision import transforms
from PIL import Image
import timm

def load_model(model_name, num_classes, model_path):
    model = timm.create_model(model_name, pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, image_path, class_names):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Check if a GPU is available and move the input tensor to the GPU if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class and probability
    _, predicted_idx = torch.max(output, 1)
    predicted_class = class_names[predicted_idx.item()]
    probability = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx.item()].item()

    print(f"Predicted class: {predicted_class}")
    print(f"Probability of prediction: {probability:.4f}")

    return predicted_class, probability

if __name__ == "__main__":
    # Specify the path to the test image
    test_image_path = 'path/to/test/image.jpg'

    # Specify the model name, number of classes, and the path to the saved model
    model_name = 'efficientnet_b0'  # Replace with the model name used during training
    num_classes = 2  # Replace with the number of classes in your dataset
    saved_model_path = 'best_model.pth'  # Replace with the path to your saved model

    # Load the model
    model = load_model(model_name, num_classes, saved_model_path)

    # Load class names (replace with your actual class names)
    class_names = ['class_0', 'class_1']

    # Make a prediction
    predicted_class, probability = predict(model, test_image_path, class_names)
