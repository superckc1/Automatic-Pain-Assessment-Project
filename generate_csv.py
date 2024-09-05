import os
import csv

# 设置数据集路径
data_dir = '/db/shared/video/UNBC-McMaster'
images_dir = os.path.join(data_dir, 'Images')
pspi_labels_dir = os.path.join(data_dir, 'Frame_Labels', 'PSPI')

# 创建新的CSV文件路径，改为你的主目录
csv_file = os.path.join('/cs/home/alykc8', 'new_pain_dataset.csv')

# 收集图像路径和对应的标签
data = []

for subject_dir in os.listdir(images_dir):
    subject_path = os.path.join(images_dir, subject_dir)
    if os.path.isdir(subject_path):
        print(f"Processing subject: {subject_dir}")
        for sequence_dir in os.listdir(subject_path):
            sequence_path = os.path.join(subject_path, sequence_dir)
            if os.path.isdir(sequence_path):
                print(f"  Processing sequence: {sequence_dir}")
                for image_file in os.listdir(sequence_path):
                    if image_file.endswith('.png'):
                        image_path = os.path.join(sequence_path, image_file)
                        # 查找对应的PSPI标签文件
                        label_file = image_file.replace('.png', '_facs.txt')
                        label_path = os.path.join(pspi_labels_dir, subject_dir, sequence_dir, label_file)
                        if os.path.exists(label_path):
                            with open(label_path, 'r') as f:
                                pspi_score = f.readline().strip()
                                data.append([image_path, pspi_score])

# 将数据写入新的CSV文件
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'pspi_score'])
    writer.writerows(data)

print(f"Generated new CSV file with {len(data)} entries at {csv_file}")
