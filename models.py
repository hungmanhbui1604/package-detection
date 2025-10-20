import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

class MobileNetV2(nn.Module):
    def __init__(self, out_dim=6):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.mobilenet.features[0][0] = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Change input channels to 4
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, out_dim)  # Adjust output layer

    def forward(self, x):
        x = self.mobilenet(x)
        return x