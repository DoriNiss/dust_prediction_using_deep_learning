# +
import torch
import torch.nn as nn
import timm.models.vision_transformer as ViT
import numpy as np

# Taken mostly from Ross Wightman, https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class ViTtoMap(nn.Module):   
    
    def __init__(self, vit, vit_decoder, output_shape=[1,81,169], patch_shape=[9,13]):
        super(ViTtoMap,self).__init__()
        self.vit = vit
        self.output_shape = output_shape
        self.patch_shape = patch_shape
        stride = [self.output_shape[1]//self.patch_shape[0], self.output_shape[2]//self.patch_shape[1]]
        self.unembed_layer = nn.ConvTranspose2d(in_channels=self.vit.embed_dim, 
                                                out_channels=self.output_shape[0], 
                                                kernel_size=self.patch_shape, stride=stride)
        self.vit_decoder = vit_decoder
        
    def forward_blocks(self,x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        if self.vit.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x
    
    def conv_decode(self,x):
        """
            Turns an embedded tensor of shape [B,num_patches,embedding_size]
            into an image of shape: [B,output_shape]
            Assuming sane input, as the ViT is the same that was used for embeddings, i.e., 
            each embedded row of size embedding_size representing a patch of shape self.patch_shape 
            e.g.: [B,117,512] -> [B,1,81,169], if patch_shape = [9,13] (9*13=117 patches of shape [9,13])
        """
        return self.unembed_layer(x)
    
    def vit_decode(self,x):
        """
            Turns an embedded tensor of shape [B,num_patches,embedding_size]
            into an image of shape: [B,output_shape]
            Assuming sane input, as the ViT is the same that was used for embeddings, i.e., 
            each embedded row of size embedding_size representing a patch of shape self.patch_shape 
            e.g.: [B,117,512] is seen as an image of  [B,1,81,169], if patch_shape = [9,13] (9*13=117 patches of shape [9,13])
        """
        return self.unembed_layer(x)

    def forward(self,x):
        x = self.forward_blocks(x)
        x = x[:,1:,:] # x[:,0,:] is the class token, so only outputs of patch embeddings are used
        B,num_patches,size_emb = x.shape
        x = x.reshape(B,self.patch_shape[0],self.patch_shape[1],size_emb)
        x = x.permute(0,3,1,2)
        return self.unembed(x)
        
