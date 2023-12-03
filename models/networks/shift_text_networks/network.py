"""
    Reference:
        - https://github.com/mlfoundations/open_clip
        - https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/encoders/modules.py
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import open_clip

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=device, pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

class CLIPTextEncoder(nn.Module):
    def __init__(self, 
                 model_name="ViT-B-32",
                 pretrained='laion2b_s34b_b79k',
                 jit=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 ):
        super().__init__()
        self.model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, jit=jit, device=device)
        
        self.device = device
        # NOTE: may need to convert clip to float precision
        # self.model = self.model.float() # turns out this is important...
        self.tokenizer = open_clip.get_tokenizer(model_name=model_name)
    
    def forward(self, x):
        tokenized_text = self.tokenizer(x).to(self.device)
        text_features = self.model.encode_text(tokenized_text).to(self.device)
        
        return text_features

class MPNetTextEncoder(nn.Module):
    def __init__(self, 
                 model_name='all-mpnet-base-v2',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 ):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(model_name).to(device)
        self.device = device
        
    def forward(self, x):
        # print(self.model.device)
        text_features = self.model.encode(x)
        
        return torch.tensor(text_features, device=self.device)
    
class PretrainedBERTTextEncoder(nn.Module):
    def __init__(self, 
                 model_name='bert-large-uncased',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 max_length = 77,
                 ):
        super().__init__()
        from transformers import BertModel, BertTokenizerFast
        
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.device = device
        self.max_length = max_length
        
    def forward(self, x):
        tokenized_text = self.tokenizer(x, truncation=True, padding=True, return_tensors="pt", 
                                        max_length=self.max_length)
        tokenized_text.to(self.device)
        text_features = self.model(**tokenized_text)[-2]
        
        return text_features