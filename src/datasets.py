# src/datasets.py
import torch
from torchvision import datasets, transforms
import numpy as np

class CIFAR10Patches:
    def __init__(self, root='data', train=True, patch_size=4, mask_ratio=0.75, batch_size=128):
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.batch_size = batch_size

        transform = transforms.Compose([
            transforms.ToTensor(),
            # normalize with CIFAR stats
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
        dataset = datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
        self.img_size = 32

    def patchify(self, imgs):
        # imgs: [B, C, H, W]
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == self.img_size and W == self.img_size
        nh = H // p
        nw = W // p
        patches = imgs.reshape(B, C, nh, p, nw, p).permute(0,2,4,1,3,5)
        patches = patches.reshape(B, nh*nw, C*p*p)  # [B, N, patch_dim]
        return patches

    def unpatchify(self, patches):
        # Not used in loader; reconstruction is done inside model
        pass

    def random_mask(self, B, N, device):
        # return mask indices of length int(N*mask_ratio)
        len_keep = int(N * (1 - self.mask_ratio))
        rand = torch.rand(B, N, device=device)
        ids = rand.argsort(dim=1)
        mask = torch.ones(B, N, device=device)
        mask[:, :len_keep] = 0  # first len_keep are kept
        # but we need shuffle order back
        return mask  # 1 means masked, 0 means keep

    def get_loader(self):
        return self.loader
