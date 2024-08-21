import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block


class Cls_Encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, norm_layer):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x



class Classifier(nn.Module):
    def __init__(self, num_classes, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.encoder = Cls_Encoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, norm_layer)
        self.embed_dim = self.encoder.cls_token.shape[-1]
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward_head(self, x):
        x = x[:, 0]  # class token
        return self.head(x)    

    def forward(self, x):
        latent = self.encoder(x)
        out = self.forward_head(latent)
        return out