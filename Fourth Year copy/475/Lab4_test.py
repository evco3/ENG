import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Yoda
from custom_dataset import custom_dataset  # Replace with your custom dataset class

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print('loading the test dataset')

# Load the test dataset
test_dataset = custom_dataset(dir='/Volumes/JACOB/ELEC475_Lab4/data/Kitti8_ROIs/test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

print('loading the model')

model = Yoda()
model.load_state_dict(torch.load('/Volumes/JACOB/ELEC475_Lab4/src/model.pth'))  # Load the trained weights
model.eval()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(device)

# print number of batches
print(f'Total number of batches: {len(test_loader)}')

# Test the model
with torch.no_grad():
    correct = 0
    total = 0

    # print every 500 batches
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        if i % 500 == 0:
            print(f'Batch number: {i} \t Predicted: {predicted} \t Actual: {labels}')
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the test images: {100 * correct / total}%')