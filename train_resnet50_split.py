import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_preprocessing_resnet50 import dataloaders
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(filename='/cs/home/alykc8/training_resnet50_split3.log', level=logging.INFO, format='%(asctime)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using device: {device}')
logging.info(f'Using device: {device}')


model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  

model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

total_samples = len(dataloaders['train'].dataset)
batch_size = dataloaders['train'].batch_size
total_batches = len(dataloaders['train'])

print(f'Total samples: {total_samples}')
print(f'Batch size: {batch_size}')
print(f'Total batches per epoch: {total_batches}')
logging.info(f'Total samples: {total_samples}')
logging.info(f'Batch size: {batch_size}')
logging.info(f'Total batches per epoch: {total_batches}')

def train(epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloaders['train']):
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # 每100个批次打印一次
            avg_loss = running_loss / 100
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {avg_loss:.4f}')
            logging.info(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {avg_loss:.4f}')
            running_loss = 0.0

    if running_loss != 0:
        avg_loss = running_loss / (i % 100 + 1)
        print(f'Epoch {epoch + 1}, Remaining Batches, Loss: {avg_loss:.4f}')
        logging.info(f'Epoch {epoch + 1}, Remaining Batches, Loss: {avg_loss:.4f}')

def validate():
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            val_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    val_loss /= len(dataloaders['val'])

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    mae = mean_absolute_error(all_labels, all_outputs)
    mse = mean_squared_error(all_labels, all_outputs)
    return val_loss, mae, mse

num_epochs = 30
best_loss = float('inf')

print("Starting the training process...")
logging.info("Starting the training process...")

for epoch in range(num_epochs):
    start_time = time.time()
    train(epoch)
    val_loss, val_mae, val_mse = validate()
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}, Validation MSE: {val_mse:.4f}, Time: {epoch_time:.2f} seconds')
    logging.info(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}, Validation MSE: {val_mse:.4f}, Time: {epoch_time:.2f} seconds')

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), '/cs/home/alykc8/best_resnet50_model3.pth')

    scheduler.step()

print("Training completed.")
logging.info("Training completed.")
