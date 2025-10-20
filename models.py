import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """
    Transformation Network for PointNet.
    
    Outputs a (K x K) transformation matrix for input of dimension K.
    
    Args:
        k (int): Input dimension (e.g., 3 for XYZ, 6 for XYZRGB, 64 for features)
    """
    def __init__(self, k):
        super().__init__()
        self.k = k

        # Shared MLP (using Conv1d for per-point transformation)
        self.conv1 = nn.Conv1d(k, 64, 1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv3 = nn.Conv1d(128, 1024, 1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialize last layer to zero (so transform ≈ identity at start)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()

    def forward(self, x):
        """
        Args:
            x: (B, K, N) - batch of point clouds or features

        Returns:
            T: (B, K, K) - transformation matrix
        """
        B = x.size(0)

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global max pooling over points (N)
        x = torch.max(x, dim=2, keepdim=True)[0]  # (B, 1024, 1)
        x = x.view(B, -1)  # (B, 1024)

        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, K*K)

        # Add identity matrix
        identity = torch.eye(self.k, device=x.device).unsqueeze(0).expand(B, -1, -1)
        T = x.view(B, self.k, self.k) + identity

        return T


class PointNet(nn.Module):
    """
    PointNet feature extractor (without final classifier).
    
    Args:
        in_channels (int): Number of input channels (3 for XYZ, 6 for XYZRGB)
        use_feature_tnet (bool): Whether to use the 64D feature T-Net (optional)
    """
    def __init__(self, in_channels, use_feature_tnet=True):
        super().__init__()
        self.in_channels = in_channels
        self.use_feature_tnet = use_feature_tnet

        # Input transform
        self.input_tnet = TNet(k=in_channels)

        # Shared MLP 1
        self.conv1 = nn.Conv1d(in_channels, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        # Feature transform (optional)
        if use_feature_tnet:
            self.feature_tnet = TNet(k=64)
        else:
            self.feature_tnet = None

        # Shared MLP 2
        self.conv2 = nn.Conv1d(64, 128, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, N) - batch of point clouds

        Returns:
            global_feat: (B, 1024) - global feature vector
        """
        B, C, N = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # Input transform
        trans_input = self.input_tnet(x)  # (B, C, C)
        x = torch.bmm(trans_input, x)     # (B, C, N)

        # First MLP
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)

        # Feature transform (optional)
        if self.use_feature_tnet:
            trans_feat = self.feature_tnet(x)  # (B, 64, 64)
            x = torch.bmm(trans_feat, x)       # (B, 64, N)

        # Second MLP
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 128, N)
        x = self.bn3(self.conv3(x))           # (B, 1024, N)

        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (B, 1024)

        return x

class MyModel(nn.Module):
    """
    Dual-branch model:
      - Image branch: MobileNetV2 (4-channel input)
      - Point cloud branch: PointNet (6-channel input)
    
    Args:
        num_classes (int): Number of output classes
    """
    def __init__(self, num_classes=6, weights=MobileNet_V2_Weights.IMAGENET1K_V1):
        super().__init__()
        
        # === Image Branch: MobileNetV2 for 4-channel input ===
        self.mobilenet = models.mobilenet_v2(weights=weights)
        # Replace first conv to accept 4 channels (e.g., RGB-D)
        self.mobilenet.features[0][0] = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Remove classifier → get 1280-dim features
        self.mobilenet.classifier = nn.Identity()
        
        # === Point Cloud Branch: PointNet for 6D input ===
        self.pointnet = PointNet(in_channels=6)
        
        # === Fusion Classifier ===
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, points):
        """
        Args:
            image:  (B, 4, H, W)  - 4-channel image (e.g., RGB-D)
            points: (B, 6, N)     - point cloud (XYZRGB)
        
        Returns:
            logits: (B, num_classes)
        """
        # Extract image features
        img_feat = self.mobilenet(image)   # (B, 1280)
        
        # Extract point cloud features
        pc_feat = self.pointnet(points)    # (B, 1024)
        
        # Concatenate features
        fused = torch.cat([img_feat, pc_feat], dim=1)  # (B, 2304)
        
        # Classify
        logits = self.classifier(fused)
        return logits