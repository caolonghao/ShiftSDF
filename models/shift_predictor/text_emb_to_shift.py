import torch
import torch.nn as nn
from .openai_model_3d import UNet3DModel

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
    
class UNetShiftPredictor(nn.Module):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = ShiftUNet(params)
        
    def forward(self, x, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # eps
        out = self.model(x, t, cond)
        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out
        
class ShiftUNet(nn.Module):
    def __init__(self, unet_params, vq_conf=None, conditioning_key=None):
        """ init method """
        super().__init__()

        self.diffusion_net = UNet3DModel(**unet_params)
        self.conditioning_key = conditioning_key # default for lsun_bedrooms


    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        # x: should be latent code. shape: (bs X z_dim X d X h X w)
        # print("c_concat: ", c_concat, "c_crossattn: ", c_crossattn)
        # print("self.conditioning_key: ", self.conditioning_key)
        if self.conditioning_key is None:
            out = self.diffusion_net(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_net(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_net(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_net(xc, t, context=cc)
            # import pdb; pdb.set_trace()
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_net(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out