import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder,EncoderLayer,Cross_EncoderLayer
from layers.Embed import Aug_Channel_Embedding,Aug_Temporal_Embedding

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.c_layer= configs.c_layer
        self.t_layer= configs.t_layer
        
        self.channel_encoder = nn.Sequential(Aug_Channel_Embedding(configs),Encoder([EncoderLayer(configs) \
                                            for _ in range(self.c_layer)])) if self.c_layer>0 else nn.Identity()
        self.temporal_encoder = nn.Sequential(Aug_Temporal_Embedding(configs),Encoder([EncoderLayer(configs) \
                                            for _ in range(self.t_layer)])) if self.t_layer>0 else nn.Identity()
        
        self.vlm_encoder = nn.Linear(768, configs.d_model)
        
        self.channel_cross_Encoder = Cross_EncoderLayer(configs)
        self.temporal_cross_Encoder = Cross_EncoderLayer(configs)
        self.projector = nn.Linear(configs.d_model,configs.num_class)

    def forward(self, x_enc, vlm_emb):
        # x_enc: [B, T, N],  vlm_emb: [B, 768]
        B,T,N=x_enc.shape
        # Encode cross channel features
        channel= self.channel_encoder(x_enc) if self.c_layer>0 else 0
        # Encode cross temporal features
        temporal= self.temporal_encoder(x_enc) if self.t_layer>0 else 0
        # Encode VLM embeddings
        vlm=self.vlm_encoder(vlm_emb).unsqueeze(1) # [B, 768] -> [B, d_model]
        
        # # Retrieval VLM emb: B N D -> B D
        vlm_c=self.channel_cross_Encoder(vlm,channel,channel).squeeze(1) if self.c_layer>0 else 0
        vlm_t=self.temporal_cross_Encoder(vlm,temporal,temporal).squeeze(1) if self.t_layer>0 else 0
        
        # B D -> B C 
        logits = self.projector(vlm_c+vlm_t)
        
        return logits 