import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification
from data_preprocessing_vit_split import train_loader, val_loader
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(filename='/cs/home/alykc8/training_vit_split_2.log', level=logging.INFO, format='%(asctime)s %(message)s')

device = torch.device("cpu")

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=1)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

total_samples = len(train_loader.dataset)
batch_size = train_loader.batch_size
total_batches = len(train_loader)

print(f'Total samples: {total_samples}')
print(f'Batch size: {batch_size}')
print(f'Total batches per epoch: {total_batches}')
logging.info(f'Total samples: {total_samples}')
logging.info(f'Batch size: {batch_size}')
logging.info(f'Total batches per epoch: {total_batches}')

def train(epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device).float()  
        labels = labels.to(device).float()  
        optimizer.zero_grad()
        outputs = model(inputs).logits.squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # 每100个批次打印一次
            avg_loss = running_loss / 100
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {avg_loss:.4f}')
            logging.info(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {avg_loss:.4f}')
            running_loss = 0.0

def validate():
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device).float() 
            labels = labels.to(device).float()  
            outputs = model(inputs).logits.squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    val_loss /= len(val_loader)

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    mae = mean_absolute_error(all_labels, all_outputs)
    mse = mean_squared_error(all_labels, all_outputs)
    return val_loss, mae, mse

num_epochs = 30  
patience = 6
best_loss = float('inf')
patience_counter = 0

print("Starting the training process...")
logging.info("Starting the training process...")

for epoch in range(num_epochs):
    train(epoch)
    val_loss, val_mae, val_mse = validate()
    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}, Validation MSE: {val_mse:.4f}')
    logging.info(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}, Validation MSE: {val_mse:.4f}')

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), '/cs/home/alykc8/best_vit_model_2.pth') 
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            logging.info("Early stopping triggered")
            break

    scheduler.step()

print("Training completed.")
logging.info("Training completed.")
