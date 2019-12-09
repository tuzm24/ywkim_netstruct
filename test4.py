import torch
model = torch.hub.load('pytorch/vision:v0.4.2', 'densenet121', pretrained=True)