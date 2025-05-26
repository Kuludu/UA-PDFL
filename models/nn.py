import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class BaseNN(nn.Module):
    def __init__(self):
        super().__init__()

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights, strict=False)

    def which_device(self):
        return next(self.parameters()).device

    def copy_gradient(self, target_model):
        for params, target_params in zip(self.parameters(), target_model.parameters()):
            if params.grad is None:
                params.grad = Variable(torch.zeros(params.size())).to(self.which_device())
            params.grad.data.zero_()
            params.grad.data.add_(params.data - target_params.data)

    def clone(self):
        clone = self.__class__().to(self.which_device())
        clone.set_weights(self.get_weights())

        return clone


class SimpleCNN(BaseNN):
    def __init__(self, out_channels=10, in_channels=3, img_size=32):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2),
                                  nn.Dropout(0.25),
                                  nn.Flatten(),
                                  nn.Linear(64 * ((img_size // 2) ** 2), 512),
                                  nn.ReLU())
        self.base = nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512, out_channels))
        self.layer_size = [864, 32, 18432, 64, 8388608, 512, 5120, 10]
        self.num_feature_extractor_layers = 6

    def forward(self, x):
        x = self.head(x)
        x = self.base(x)

        return x
