import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class PainDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['pspi_score']

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

csv_file = '/cs/home/alykc8/data/new_pain_dataset_with_subject_id.csv'
data_frame = pd.read_csv(csv_file)

print(f"Data preprocessing completed. Total samples: {len(data_frame)}")

subject_ids = data_frame['subject_id'].unique()

# 创建数据加载器（示例，用于后续训练脚本）
dataset = PainDataset(dataframe=data_frame, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
