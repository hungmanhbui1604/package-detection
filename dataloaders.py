import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
import os
import open3d as o3d


class PackageTrainDataset(Dataset):
    def __init__(
        self,
        df,
        rgb_dir,
        depth_dir,
        ply_dir,
        roi,
        num_points=1024,
        rgb_transform=None,
        depth_transform=None,
        ply_transform=False,
    ):
        self.df = df
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.ply_dir = ply_dir
        self.roi = roi  # Region of interest for cropping images
        self.num_points = num_points  # Fixed number of points per cloud
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.ply_transform = ply_transform

    def __len__(self):
        return len(self.df)

    def _uniform_sample(self, points, colors, num_points):
        """Randomly sample `num_points` from the point cloud (with replacement if needed)."""
        N = points.shape[0]
        if N == 0:
            return np.zeros((num_points, 3)), np.zeros((num_points, 3))
        if N >= num_points:
            # Random subsample
            indices = np.random.choice(N, num_points, replace=False)
        else:
            # Oversample with replacement
            indices = np.random.choice(N, num_points, replace=True)
        return points[indices], colors[indices]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_prefix = self.df.iloc[idx, 0]
        img_name = img_name_prefix[6:]  # Remove "image_" prefix
        rgb_path = os.path.join(self.rgb_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)
        ply_path = os.path.join(self.ply_dir, img_name[:-3] + 'ply')

        # Load and crop RGB + Depth
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        rgb = rgb.crop(self.roi)
        depth = depth.crop(self.roi)
        rgb = np.array(rgb)  # (H, W, 3)
        depth = np.array(depth)  # (H, W)

        # Load point cloud
        ply = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(ply.points)  # (N, 3)
        colors = np.asarray(ply.colors)  # (N, 3) — in [0, 1]

        # Handle empty point clouds
        if len(points) == 0:
            points = np.zeros((1, 3))
            colors = np.zeros((1, 3))

        # Uniformly sample to fixed number of points
        points, colors = self._uniform_sample(points, colors, self.num_points)

        # Apply transforms or convert to tensor
        if self.rgb_transform:
            rgb = self.rgb_transform(rgb)
        else:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()

        if self.depth_transform:
            depth = self.depth_transform(depth)
        else:
            depth = torch.from_numpy(depth).unsqueeze(0).float()

        rgb_depth = torch.cat((rgb, depth), dim=0)  # (4, H, W)

        # Point cloud: convert to (6, num_points)
        points = torch.from_numpy(points).transpose(0, 1).float()  # (3, N)
        colors = torch.from_numpy(colors).transpose(0, 1).float()  # (3, N)
        # Apply transforms
        if self.ply_transform:
            mean_rgb = torch.tensor([0.485, 0.456, 0.406]).view(3, 1)
            std_rgb = torch.tensor([0.229, 0.224, 0.225]).view(3, 1)
            colors = (colors - mean_rgb) / std_rgb

        xyz_rgb = torch.cat((points, colors), dim=0)  # (6, N)

        # Labels
        labels = self.df.iloc[idx, 1:].to_numpy().astype(np.float32)
        labels = torch.from_numpy(labels)

        return {
            'names': img_name_prefix,
            'rgb_depths': rgb_depth,      # (4, H, W)
            'xyz_rgbs': xyz_rgb,          # (6, num_points)
            'labels': labels              # (num_labels,)
        }

class PackageTestDataset(Dataset):
    def __init__(
        self,
        rgb_dir,
        depth_dir,
        ply_dir,
        roi,
        num_points=32768,
        rgb_transform=None,
        depth_transform=None,
        ply_transform=False,
    ):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.ply_dir = ply_dir
        self.roi = roi  # Region of interest for cropping images
        self.num_points = num_points  # Fixed number of points per cloud
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.ply_transform = ply_transform

    def __len__(self):
        return len(os.listdir(self.rgb_dir))

    def _uniform_sample(self, points, colors, num_points):
        """Randomly sample `num_points` from the point cloud (with replacement if needed)."""
        N = points.shape[0]
        if N == 0:
            return np.zeros((num_points, 3)), np.zeros((num_points, 3))
        if N >= num_points:
            # Random subsample
            indices = np.random.choice(N, num_points, replace=False)
        else:
            # Oversample with replacement
            indices = np.random.choice(N, num_points, replace=True)
        return points[indices], colors[indices]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.listdir(self.rgb_dir)[idx]
        img_name_prefix = 'image_' + img_name
        rgb_path = os.path.join(self.rgb_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)
        ply_path = os.path.join(self.ply_dir, img_name[:-3] + 'ply')

        # Load and crop RGB + Depth
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        rgb = rgb.crop(self.roi)
        depth = depth.crop(self.roi)
        rgb = np.array(rgb)  # (H, W, 3)
        depth = np.array(depth)  # (H, W)

        # Load point cloud
        ply = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(ply.points)  # (N, 3)
        colors = np.asarray(ply.colors)  # (N, 3) — in [0, 1]

        # Handle empty point clouds
        if len(points) == 0:
            points = np.zeros((1, 3))
            colors = np.zeros((1, 3))

        # Uniformly sample to fixed number of points
        points, colors = self._uniform_sample(points, colors, self.num_points)

        # Apply transforms or convert to tensor
        if self.rgb_transform:
            rgb = self.rgb_transform(rgb)
        else:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()

        if self.depth_transform:
            depth = self.depth_transform(depth)
        else:
            depth = torch.from_numpy(depth).unsqueeze(0).float()

        rgb_depth = torch.cat((rgb, depth), dim=0)  # (4, H, W)

        # Point cloud: convert to (6, num_points)
        points = torch.from_numpy(points).transpose(0, 1).float()  # (3, N)
        colors = torch.from_numpy(colors).transpose(0, 1).float()  # (3, N)
        # Apply transforms
        if self.ply_transform:
            mean_rgb = torch.tensor([0.485, 0.456, 0.406]).view(3, 1)
            std_rgb = torch.tensor([0.229, 0.224, 0.225]).view(3, 1)
            colors = (colors - mean_rgb) / std_rgb

        xyz_rgb = torch.cat((points, colors), dim=0)  # (6, N)

        return {
            'names': img_name_prefix,
            'rgb_depths': rgb_depth,      # (4, H, W)
            'xyz_rgbs': xyz_rgb,          # (6, num_points)
        }