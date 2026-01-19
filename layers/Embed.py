import math
import torch
import random
import torch.nn as nn
from layers.Augmentation import get_augmentation


class Aug_Channel_Embedding(nn.Module):
    def __init__(self,configs,seq_len=None):
        super().__init__()
        seq_len= configs.seq_len if seq_len is None else seq_len
        aug_idxs = configs.augmentations.split(",")
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in aug_idxs]
        )
        self.Channel_Embedding =nn.Linear(seq_len, configs.d_model)
        self.pos_emb = PositionalEmbedding(d_model=configs.d_model)

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        B,T,N= x.shape
        x = x.transpose(1,2)  # (batch_size, enc_in, seq_len)
        aug_idx = random.randint(0, len(self.augmentation) - 1)
        x_aug = self.augmentation[aug_idx](x)
        emb=self.Channel_Embedding(x_aug)
        return emb+self.pos_emb(emb)
    
class Aug_Temporal_Embedding(nn.Module):
    def __init__(self,configs,seq_len=None):
        super().__init__()
        seq_len= configs.seq_len if seq_len is None else seq_len
        self.patch_len = configs.patch_len
        aug_idxs = configs.augmentations.split(",")
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in aug_idxs]
        )
        self.Temporal_Embedding =CrossChannelPatching(configs) if self.patch_len >1 else nn.Linear(configs.enc_in, configs.d_model)
        self.pos_emb = PositionalEmbedding(d_model=configs.d_model)

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        B,T,N= x.shape
        x = x.transpose(1,2)  # (batch_size, enc_in, seq_len)
        aug_idx = random.randint(0, len(self.augmentation) - 1)
        x_aug = self.augmentation[aug_idx](x)
        if self.patch_len ==1: x_aug=x_aug.transpose(1,2)
        emb=self.Temporal_Embedding(x_aug)
        return emb+self.pos_emb(emb)
    
class CrossChannelPatching(nn.Module):
    def __init__(self, configs):
        super().__init__()
        patch_len=configs.patch_len
        stride = configs.patch_len
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=configs.d_model,
            kernel_size=(configs.enc_in, patch_len),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        self.padding = nn.ReplicationPad1d((0, stride))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x=self.padding(x).unsqueeze(1)
        x = self.tokenConv(x).squeeze(2).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, learnable=True,max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  

        if learnable:
            self.pe = nn.Parameter(pe, requires_grad=True)
        else:
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)]
