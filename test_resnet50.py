import time
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import logging
from data_preprocessing_resnet50 import dataloaders

logging.basicConfig(filename='/cs/home/alykc8/testing_resnet50.log', level=logging.INFO, format='%(asctime)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

model = model.to(device)
model.load_state_dict(torch.load('/cs/home/alykc8/best_resnet50_model3.pth'))

criterion = nn.MSELoss()

def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc

def test_model():
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            test_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    test_loss /= len(dataloaders['test'])

    all_labels = np.concatenate(all_labels).flatten()
    all_outputs = np.concatenate(all_outputs).flatten()

    mae = mean_absolute_error(all_labels, all_outputs)
    mse = mean_squared_error(all_labels, all_outputs)

    pcc, _ = pearsonr(all_labels, all_outputs)

    ccc = concordance_correlation_coefficient(all_labels, all_outputs)

    return test_loss, mae, mse, pcc, ccc

test_loss, test_mae, test_mse, test_pcc, test_ccc = test_model()

print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}, PCC: {test_pcc:.4f}, CCC: {test_ccc:.4f}')
logging.info(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}, PCC: {test_pcc:.4f}, CCC: {test_ccc:.4f}')
