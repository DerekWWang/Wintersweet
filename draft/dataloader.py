import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from timm.data import resolve_model_data_config, create_transform
import os

class ImageBagDataset(Dataset):
    def __init__(self, csv_file, train=True, model=None):
        self.data = pd.read_csv(csv_file)
        if train:
            self.path = "C:\\Code\\DL\\bbosis\\data\\small\\bags\\train\\"
        else:
            self.path = "C:\\Code\\DL\\bbosis\\data\\small\\bags\\test\\"
        self.slide_ids = self.data["image_id"].astype(str).tolist()
        self.labels = (self.data["label"]).astype(np.int64).to_numpy()

        if model is not None:
            cfg = resolve_model_data_config(model)
            self.transform = create_transform(**cfg, is_training=train)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bag_dir = os.path.join(self.path, self.slide_ids[idx])
        bag_tensors = []
        for img_name in sorted(os.listdir(bag_dir)):
            img_path = os.path.join(bag_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            bag_tensors.append(self.transform(img))
        bag_tensor = torch.stack(bag_tensors, dim=0)   # [K,3,H,W]
        return bag_tensor, int(self.labels[idx]), self.slide_ids[idx]