import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import timm
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np

def build_model(model_name, num_classes, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    model = model.to(device)
    
    # Modify the classifier head for the number of classes in your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Load data
data_dir = 'path/to/your/data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Choose the model architecture (EfficientNet or other models from timm)
model_name = 'efficientnet_b0'  # Replace with the desired timm model name
model = build_model(model_name, len(class_names), pretrained=True)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().detach().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            

            if phase == 'val':
                # Calculate ROC AUC score for each class
                for i in range(len(class_names)):
                    class_auc = roc_auc_score(all_labels, all_preds[:, i])
                    print('AUC for Class {}: {:.4f}'.format(class_names[i], class_auc))

                # Store the model with the best AUC on validation set
                if epoch_acc > best_auc:
                    best_auc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    # Save the best model
    torch.save(best_model_wts, 'best_model.pth')

# Train the model
train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
