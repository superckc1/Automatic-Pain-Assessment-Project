import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification
from sklearn.model_selection import KFold
import pandas as pd
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader 
from data_preprocessing_vit_cv import PainDataset, transform

logging.basicConfig(filename='/cs/home/alykc8/vit_cv_training3.log', level=logging.INFO, format='%(asctime)s %(message)s')

csv_file = '/cs/home/alykc8/data/new_pain_dataset_with_subject_id.csv'
data_frame = pd.read_csv(csv_file)
subject_ids = data_frame['subject_id'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def undersample(df):
    no_pain_df = df[df['pspi_score'] == 0]
    pain_df = df[df['pspi_score'] != 0]
    max_non_zero_pain_count = pain_df['pspi_score'].value_counts().max()
    undersampled_no_pain_df = no_pain_df.sample(n=max_non_zero_pain_count * 2, random_state=42)
    undersampled_train_df = pd.concat([undersampled_no_pain_df, pain_df])
    undersampled_train_df = undersampled_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return undersampled_train_df

fold = 0
best_loss = float('inf')

for train_index, val_index in kf.split(subject_ids):
    fold += 1
    print(f'Starting fold {fold}')
    logging.info(f'Starting fold {fold}')

    train_ids, val_ids = subject_ids[train_index], subject_ids[val_index]

    train_df = data_frame[data_frame['subject_id'].isin(train_ids)]
    val_df = data_frame[data_frame['subject_id'].isin(val_ids)]

    undersampled_train_df = undersample(train_df)

    print(f"Fold {fold}, after undersampling: {len(undersampled_train_df)} samples")
    logging.info(f"Fold {fold}, after undersampling: {len(undersampled_train_df)} samples")

    train_dataset = PainDataset(dataframe=undersampled_train_df, transform=transform)
    val_dataset = PainDataset(dataframe=val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    logging.info(f'Using device: {device}')

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=1)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    def train_epoch():
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
        return running_loss / len(train_loader)

    def validate_epoch():
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

    num_epochs = 11
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_epoch()
        val_loss, val_mae, val_mse = validate_epoch()
        epoch_time = time.time() - start_time

        print(f'Fold {fold}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}, Validation MSE: {val_mse:.4f}, Time: {epoch_time:.2f} seconds')
        logging.info(f'Fold {fold}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}, Validation MSE: {val_mse:.4f}, Time: {epoch_time:.2f} seconds')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'/cs/home/alykc8/models/best_vit_model_fold_{fold}.pth')

        scheduler.step()

print("Cross-validation training completed.")
logging.info("Cross-validation training completed.")
