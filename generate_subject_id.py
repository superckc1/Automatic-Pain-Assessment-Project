import pandas as pd

csv_file = '/cs/home/alykc8/new_pain_dataset.csv'
data_frame = pd.read_csv(csv_file)

data_frame['subject_id'] = data_frame['image_path'].apply(lambda x: x.split('/')[-3])

data_frame.to_csv('/cs/home/alykc8/new_pain_dataset_with_subject_id.csv', index=False)

print("Subject IDs added and saved successfully.")
