import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
import os


class PackageTrainDataset(Dataset):
    def __init__(self, df, rgb_dir, depth_dir, roi, rgb_transform=None, depth_transform=None):
        self.df = df
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.roi = roi
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name_prefix = self.df.iloc[idx, 0]
        img_name = img_name_prefix[6:] # Remove "image_" prefix
        rgb_path, depth_path = os.path.join(self.rgb_dir, img_name), os.path.join(self.depth_dir, img_name)
        rgb, depth = Image.open(rgb_path).convert('RGB'), Image.open(depth_path).convert('L')
        rgb, depth = rgb.crop(self.roi), depth.crop(self.roi)
        rgb, depth = np.array(rgb), np.array(depth)
        if self.rgb_transform:
            rgb = self.rgb_transform(rgb)
        else:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() # Convert to torch tensor and change to CxHxW
        if self.depth_transform:
            depth = self.depth_transform(depth)
        else:
            depth = torch.from_numpy(depth).unsqueeze(0).float() # Add channel dimension
        inputs = torch.cat((rgb, depth), dim=0) # Concatenate along channel dimension

        labels = self.df.iloc[idx, 1:].to_numpy().astype(np.float32)
        labels = torch.from_numpy(labels)
        
        return {'names': img_name_prefix, 'inputs': inputs, 'labels': labels}

class PackageTestDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, roi, rgb_transform=None, depth_transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.roi = roi
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
    
    def __len__(self):
        return len(os.listdir(self.rgb_dir))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.listdir(self.rgb_dir)[idx]
        img_name_prefix = 'image_' + img_name
        rgb_path, depth_path = os.path.join(self.rgb_dir, img_name), os.path.join(self.depth_dir, img_name)
        rgb, depth = Image.open(rgb_path).convert('RGB'), Image.open(depth_path).convert('L')
        rgb, depth = rgb.crop(self.roi), depth.crop(self.roi)
        rgb, depth = np.array(rgb), np.array(depth)
        if self.rgb_transform:
            rgb = self.rgb_transform(rgb)
        else:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        if self.depth_transform:
            depth = self.depth_transform(depth)
        else:
            depth = torch.from_numpy(depth).unsqueeze(0).float()
        inputs = torch.cat((rgb, depth), dim=0)

        return {'names': img_name_prefix, 'inputs': inputs}