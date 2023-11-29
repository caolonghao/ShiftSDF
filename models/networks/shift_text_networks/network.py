"""
    Reference:
        - https://github.com/mlfoundations/open_clip
"""

import torch
import torch.nn as nn

import open_clip

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