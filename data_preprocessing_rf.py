import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PainDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224, 224))  
        image = np.array(image).flatten()  
        label = self.dataframe.iloc[idx]['pspi_score']
        return image, label

csv_file = '/cs/home/alykc8/data/new_pain_dataset_with_subject_id.csv'
data_frame = pd.read_csv(csv_file)

print(f"Data preprocessing completed. Total samples: {len(data_frame)}")

subject_ids = data_frame['subject_id'].unique()
