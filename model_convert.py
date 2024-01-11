import torch
from torchvision.models import efficientnet_b0
import timm
import torch.nn as nn

model_name = 'efficientnet_b0'  # You can choose other variants (b1, b2, ..., b7)
model = timm.create_model(model_name, pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, out_features=11)
model=torch.load(r'EffNetB0_10_99_90_LRS.pth', map_location=torch.device('cpu'))
torch.save(model, 'EffNetB0_10_99_90_LRS.pt')
