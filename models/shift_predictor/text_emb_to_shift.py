import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class TextShiftPredictor(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = config["input_dim"]
        self.image_channel = config["image_channel"] # the channel of latent space
        self.image_size = config["image_size"] # the dimention of latent space
        self.cond_model = config["cond_model"]
        
        # self.predictor = nn.Linear(self.input_dim, self.image_channel * self.image_size * self.image_size * self.image_size)
        
        self.predictor = nn.Sequential(
            nn.Linear(self.input_dim, 32 * 4 * 4 * 4),
            Swish(),
            View((-1, 32, 4, 4, 4)),
            nn.ConvTranspose3d(32, 16, 4, 2, 1), # 8
            Swish(),
            nn.ConvTranspose3d(16, 16, 1, 1, 0), # 8
            Swish(),
            nn.ConvTranspose3d(16, self.image_channel, 4, 2, 1) # 16
        )
        
        # self.predictor = nn.Sequential(
        #     nn.Linear(77 * 1280, 512 * 4 * 4 * 4),
        #     Swish(),
        #     View((-1, 512, 4, 4, 4)),
        #     nn.ConvTranspose3d(512, 256, 4, 2, 1), # 8
        #     Swish(),
        #     nn.ConvTranspose3d(256, 64, 1, 1, 0), # 8
        #     Swish(),
        #     nn.ConvTranspose3d(64, self.image_channel, 4, 2, 1) # 16
        # )
        
    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, -1).contiguous()
        return self.predictor(x).reshape(-1, self.image_channel, self.image_size, self.image_size, self.image_size)