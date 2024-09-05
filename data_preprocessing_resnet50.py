import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PainDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

train_csv = '/cs/home/alykc8/vit_undersampled_train_dataset.csv'
val_csv = '/cs/home/alykc8/vit_val_dataset.csv'
test_csv = '/cs/home/alykc8/vit_test_dataset.csv'

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

train_dataset = PainDataset(dataframe=train_df, transform=transform)
val_dataset = PainDataset(dataframe=val_df, transform=transform)
test_dataset = PainDataset(dataframe=test_df, transform=transform)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8),
    'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
}

print("Data preprocessing completed. Total samples: {}".format(len(train_dataset)))
