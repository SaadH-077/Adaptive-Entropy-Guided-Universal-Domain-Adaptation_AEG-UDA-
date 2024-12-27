from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable

# Feature extractor which is basically a modified ResNet50 model
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()
        self.pretrained = pretrained

        # If pretrained is True, load the latest weights of the ResNet50 model
        if pretrained:
            self.resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.resnet = resnet50()
        
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        x = self.resnet(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        return x

# Classifier head to pair with ResNet
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=31, input_size=2048, temp=0.05):
        super(ResNetClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if return_feat:
            return x
        x = F.normalize(x)
        x = self.fc(x)/self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self, m):
        m.weight.data.normal_(0.0, 0.1)