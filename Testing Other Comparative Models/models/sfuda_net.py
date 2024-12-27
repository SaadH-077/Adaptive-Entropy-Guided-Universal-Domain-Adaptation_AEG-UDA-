from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable
from config import config_supervised

Fs_dims = config_supervised.settings['Fs_dims']
cnn_to_use = config_supervised.settings['cnn_to_use']

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

class modnet(nn.Module):
    
    def __init__(self, num_C, num_Cs_dash, num_Ct_dash, cnn=cnn_to_use, additional_components=[]): 
        
        super(modnet, self).__init__()

        # Frozen initial conv layers
        if cnn=='resnet50':
            self.M = ResNetFeatureExtractor()
        else:
            raise NotImplementedError('Not implemented for ' + str(cnn))

        self.Fs = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ELU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024,Fs_dims),
            nn.ELU(),
            nn.Linear(Fs_dims, Fs_dims),
            nn.BatchNorm1d(Fs_dims),
            nn.ELU(),
        )

        self.Ft = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ELU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024,Fs_dims),
            nn.ELU(),
            nn.Linear(Fs_dims, Fs_dims),
            nn.BatchNorm1d(Fs_dims),
            nn.ELU(),
        )

        self.G = nn.Sequential(
            nn.Linear(Fs_dims,1024),
            nn.ELU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024,2048),
            nn.ELU(),
            nn.Linear(2048, 2048),
        )

        self.Cs = nn.Sequential(
            nn.Linear(Fs_dims, num_C + num_Cs_dash)
        )
        
        # Negative class classifier. Change this to vary the size of the negative class classifier.
        n = config_supervised.settings['num_C'] + config_supervised.settings['num_Cs_dash']
        num_negative_classes = int(n*(n-1)/2)
        # num_negative_classes = 150

        self.Cn = nn.Sequential(
            nn.Linear(Fs_dims, num_negative_classes)
        )

        self.components = {
            'M': self.M,
            'Fs': self.Fs,
            'Ft': self.Ft,
            'G': self.G,
            'Cs': self.Cs,
            'Cn': self.Cn,
        }

    def forward(self, x, which_fext='original'):
        raise NotImplementedError('Implemented a custom forward in train loop')


def no_param(model):
    param = 0
    for p in list(model.parameters()):
        n=1
        for i in list(p.size()):
            n*= i
        param += n
    return param


if __name__=='__main__':
    raise NotImplementedError('Please check README.md for execution details')