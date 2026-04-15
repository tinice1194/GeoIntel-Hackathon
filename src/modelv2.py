import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabModel(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()

        
        if pretrained:
            self.model = deeplabv3_resnet50(weights="DEFAULT")
        else:
            self.model = deeplabv3_resnet50(weights=None)

        
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        
        return self.model(x)


def load_model(checkpoint_path, num_classes=11, device="cpu"):
    model = DeepLabModel(num_classes=num_classes)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model
