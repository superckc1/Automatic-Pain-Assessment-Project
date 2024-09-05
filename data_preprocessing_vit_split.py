import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor

class PainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]
        if not os.path.isabs(img_path):
            img_path = os.path.join('/db/shared/video/UNBC-McMaster', img_path)
        image = Image.open(img_path).convert("RGB")
        label = float(self.data_frame.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

def transform(image):
    return feature_extractor(images=image, return_tensors="pt").pixel_values[0]

train_dataset = PainDataset(csv_file='/cs/home/alykc8/vit_undersampled_train_dataset.csv', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

val_dataset = PainDataset(csv_file='/cs/home/alykc8/vit_val_dataset.csv', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

test_dataset = PainDataset(csv_file='/cs/home/alykc8/vit_test_dataset.csv', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
