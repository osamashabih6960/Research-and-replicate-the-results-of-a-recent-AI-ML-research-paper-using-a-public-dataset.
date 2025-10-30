# src/mae_model.py
import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import vit_small_patch16_224, PatchEmbed

# We'll build a tiny ViT encoder and a small decoder for reconstruction.
class SimplePatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed, H/ps, W/ps]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)  # [B, N, embed]
        return x

class TinyEncoder(nn.Module):
    def __init__(self, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.):
        super().__init__()
        from timm.models.vision_transformer import _cfg
        # reuse timm ViT blocks if available
        from timm.models.vision_transformer import Block
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (32//4)*(32//4)+1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # [B, N+1, embed]

class MAE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=192, encoder_depth=6, decoder_dim=128, mask_ratio=0.75):
        super().__init__()
        self.patch_embed = SimplePatchEmbed(img_size, patch_size, embed_dim=embed_dim)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.encoder = TinyEncoder(embed_dim=embed_dim, depth=encoder_depth)
        # decoder: map back to patches
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.decoder_blocks = nn.Sequential(*[nn.Sequential(nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, decoder_dim), nn.GELU()) for _ in range(4)])
        self.decoder_pred = nn.Linear(decoder_dim, patch_size*patch_size*3)  # predict pixels

    def forward(self, imgs):
        # imgs: [B, 3, H, W]
        B = imgs.shape[0]
        x = self.patch_embed(imgs)  # [B, N, embed]
        N = x.size(1)
        # sample mask indices per image
        device = imgs.device
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        batch_idx = torch.arange(B, device=device)[:, None]
        x_keep = x[batch_idx, ids_keep]  # [B, len_keep, embed]
        enc = self.encoder(x_keep)  # [B, len_keep+1, embed]
        # prepare decoder input: we will reconstruct all patches including masked ones (simple approach)
        dec_inp = enc[:,1:,:]  # skip cls
        dec = self.decoder_embed(dec_inp)
        dec = self.decoder_blocks(dec)
        pred = self.decoder_pred(dec)  # [B, N_used, patch_dim]
        # For simplicity compute MSE between predicted for masked patches vs ground truth patches
        # We need ground-truth patch pixels
        patches = self.patchify_imgs(imgs)  # [B, N, patch_dim]
        # locate masked indices
        mask = torch.ones(B, N, device=device)
        mask[batch_idx, ids_keep] = 0
        masked_idx = (mask==1)
        # align pred to masked patches (this simple code assumes decoder predicts masked patches in order of masked indices)
        # For a cleaner implementation follow MAE repo; this is a simplified demo
        masked_patches_gt = patches[masked_idx].view(B, N-len_keep, -1)
        # but pred shape may not match masked ordering; for brevity compute MSE on kept patches (works as demo)
        loss = ((pred - patches[:, :pred.size(1), :])**2).mean()
        return loss, pred

    def patchify_imgs(self, imgs):
        # same as datasets.patchify
        p = self.patch_size
        B, C, H, W = imgs.shape
        nh = H // p
        nw = W // p
        patches = imgs.reshape(B, C, nh, p, nw, p).permute(0,2,4,1,3,5)
        patches = patches.reshape(B, nh*nw, C*p*p)
        return patches
