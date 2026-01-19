import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)
    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x= encoder_layer(x)
        return x
    
    
class EncoderLayer(nn.Module):
    def __init__(self, configs,dim=None):
        super().__init__()
        d_model = configs.d_model if dim is None else dim
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, configs.n_heads)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(2*d_model)),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(int(2*d_model), d_model),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        
        x_att = self.attn(x)
        x =self.norm1(x + x_att)
        
        x_ln = self.ffn(x)
        x = self.norm2(x + x_ln)
        
        return x


class Cross_EncoderLayer(nn.Module):
    def __init__(self, configs,dim=None):
        super().__init__()
        d_model = configs.d_model if dim is None else dim
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = Cross_Attention(d_model,configs.n_heads)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(2*d_model)),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(int(2*d_model), d_model),
            nn.Dropout(configs.dropout)
        )

    def forward(self, q,k,v):
        
        x_att = self.cross_attn(q, k, v)
        x =self.norm1(q + x_att)
        
        x_ln = self.ffn(x)
        x = self.norm2(x + x_ln)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads, init=0):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = (dim // n_heads) ** -0.5 
        if init:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, n, dim = x.shape
        n_heads = self.n_heads
        head_dim = dim // n_heads

        qkv = self.qkv(x).reshape(b, n, 3, n_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = F.softmax(attn, dim=-1)  

        out = attn @ v  
        out = out.transpose(1, 2).reshape(b, n, dim)  

        return self.out_proj(out)
    
    
class Cross_Attention(nn.Module):
    def __init__(self, dim, n_heads, init=0):
        super().__init__()
        self.n_heads = n_heads
        self.dim= dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = (dim // n_heads) ** -0.5 
        if init:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, q,k,v):
        b,n, dim = q.shape
        n_heads = self.n_heads
        head_dim = self.dim // self.n_heads
        
        q= self.q_proj(q).reshape(q.shape[0], q.shape[1], n_heads, head_dim).permute(0, 2, 1, 3)
        k= self.k_proj(k).reshape(k.shape[0], k.shape[1], n_heads, head_dim).permute(0, 2, 1, 3)
        v= self.v_proj(v).reshape(v.shape[0], v.shape[1], n_heads, head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = F.softmax(attn, dim=-1)  

        out = attn @ v  
        out = out.transpose(1, 2).reshape(b, n, dim)  

        return self.out_proj(out)