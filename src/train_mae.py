# src/train_mae.py
import torch, os, argparse
from src.datasets import CIFAR10Patches
from src.mae_model import MAE
from torch.optim import Adam

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = CIFAR10Patches(root='data', train=True, patch_size=args.patch_size, mask_ratio=args.mask_ratio, batch_size=args.batch_size)
    loader = data.get_loader()
    model = MAE(img_size=32, patch_size=args.patch_size, embed_dim=args.embed_dim, encoder_depth=args.depth, mask_ratio=args.mask_ratio)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            loss, _ = model(imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            if i % 50 == 0:
                print(f"Epoch {epoch} Iter {i} Loss {running_loss/(i+1):.4f}")
        print(f"Epoch {epoch} mean loss: {running_loss/len(loader):.4f}")
        # save checkpoint
        os.makedirs('results', exist_ok=True)
        torch.save(model.state_dict(), f"results/mae_epoch{epoch}.pth")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
