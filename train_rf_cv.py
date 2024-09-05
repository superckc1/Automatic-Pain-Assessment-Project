import time
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import logging
from data_preprocessing_svm import PainDataset
from joblib import dump

logging.basicConfig(filename='/cs/home/alykc8/rf_cv_training3.log', level=logging.INFO, format='%(asctime)s %(message)s')

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

    print(f"After undersampling: {len(undersampled_train_df)} samples")
    logging.info(f"After undersampling: {len(undersampled_train_df)} samples")

    return undersampled_train_df

def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc

fold = 0
all_train_losses, all_val_losses = [], []
all_val_mae, all_val_mse = [], []
all_val_pcc, all_val_ccc = [], []

for train_index, val_index in kf.split(subject_ids):
    fold += 1
    print(f'Starting fold {fold}')
    logging.info(f'Starting fold {fold}')

    train_ids, val_ids = subject_ids[train_index], subject_ids[val_index]

    train_df = data_frame[data_frame['subject_id'].isin(train_ids)]
    val_df = data_frame[data_frame['subject_id'].isin(val_ids)]

    undersampled_train_df = undersample(train_df)
    train_dataset = PainDataset(dataframe=undersampled_train_df)
    val_dataset = PainDataset(dataframe=val_df)

    train_data, train_labels = zip(*[train_dataset[i] for i in range(len(train_dataset))])
    val_data, val_labels = zip(*[val_dataset[i] for i in range(len(val_dataset))])

    train_data = np.array(train_data)
    val_data = np.array(val_data)
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    train_data = train_data.reshape(train_data.shape[0], -1)
    val_data = val_data.reshape(val_data.shape[0], -1)

    batch_size = 16
    num_batches = len(train_data) // batch_size

    batch_train_losses = []
    for i in range(num_batches):
        batch_data = train_data[i * batch_size:(i + 1) * batch_size]
        batch_labels = train_labels[i * batch_size:(i + 1) * batch_size]

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(batch_data, batch_labels)

        batch_train_predictions = rf_model.predict(batch_data)
        batch_train_loss = mean_squared_error(batch_labels, batch_train_predictions)
        batch_train_losses.append(batch_train_loss)

    train_loss = np.mean(batch_train_losses)
    all_train_losses.append(train_loss)
    print(f'Fold {fold}, Train Loss: {train_loss:.4f}')
    logging.info(f'Fold {fold}, Train Loss: {train_loss:.4f}')

    val_predictions = rf_model.predict(val_data)
    val_loss = mean_squared_error(val_labels, val_predictions)
    all_val_losses.append(val_loss)
    print(f'Fold {fold}, Validation Loss: {val_loss:.4f}')
    logging.info(f'Fold {fold}, Validation Loss: {val_loss:.4f}')

    val_mae = mean_absolute_error(val_labels, val_predictions)
    val_mse = val_loss
    all_val_mae.append(val_mae)
    all_val_mse.append(val_mse)

    val_pcc, _ = pearsonr(val_labels, val_predictions)
    all_val_pcc.append(val_pcc)

    val_ccc = concordance_correlation_coefficient(val_labels, val_predictions)
    all_val_ccc.append(val_ccc)

    print(f'Fold {fold}, Validation MAE: {val_mae:.4f}, Validation MSE: {val_mse:.4f}, PCC: {val_pcc:.4f}, CCC: {val_ccc:.4f}')
    logging.info(f'Fold {fold}, Validation MAE: {val_mae:.4f}, Validation MSE: {val_mse:.4f}, PCC: {val_pcc:.4f}, CCC: {val_ccc:.4f}')

print("Cross-validation training completed.")
logging.info("Cross-validation training completed.")
