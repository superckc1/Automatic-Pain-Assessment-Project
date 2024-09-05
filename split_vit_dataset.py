import pandas as pd
from sklearn.model_selection import train_test_split

csv_file = '/cs/home/alykc8/data/new_pain_dataset_with_subject_id.csv'
data_frame = pd.read_csv(csv_file)

if 'subject_id' not in data_frame.columns:
    raise ValueError("The dataset must contain a 'subject_id' column.")

subject_ids = data_frame['subject_id'].unique()
train_ids, test_val_ids = train_test_split(subject_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42)

train_df = data_frame[data_frame['subject_id'].isin(train_ids)]
val_df = data_frame[data_frame['subject_id'].isin(val_ids)]
test_df = data_frame[data_frame['subject_id'].isin(test_ids)]

train_df.to_csv('/cs/home/alykc8/vit_train_dataset.csv', index=False)
val_df.to_csv('/cs/home/alykc8/vit_val_dataset.csv', index=False)
test_df.to_csv('/cs/home/alykc8/vit_test_dataset.csv', index=False)

print("Dataset split and saved successfully without under-sampling, using new_pain_dataset_with_subject_id.csv.")
