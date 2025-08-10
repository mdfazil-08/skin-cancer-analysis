import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Paths
csv_path = '/home/fazil/Projects/cv-projects/labels.csv'
image_dir = '/home/fazil/Projects/cv-projects/images/'
output_base = '/home/fazil/Projects/cv-projects/dataset/'  # where train/val/test will go

# Read CSV
df = pd.read_csv(csv_path)

# Add full image path and file extension if needed
df['filename'] = df['image_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

# Split into train, val, test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.1, stratify=train_df['dx'], random_state=42)

splits = {'train': train_df, 'val': val_df, 'test': test_df}

# Move files into folders
for split_name, split_df in splits.items():
    for _, row in split_df.iterrows():
        class_dir = os.path.join(output_base, split_name, row['dx'])
        os.makedirs(class_dir, exist_ok=True)
        shutil.copy(row['filename'], class_dir)
