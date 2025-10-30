# src/linear_probe.py
import torch, argparse
from src.datasets import CIFAR10Patches
from src.mae_model import MAE
import torch.nn as nn
from torch.optim import SGD
from torchvision import datasets, transforms

def extract_features(model, loader, device):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            # get patch embeddings, run encoder, and take cls token or mean pooling
            x = model.patch_embed(imgs)
            enc = model.encoder(x)
            feat = enc[:,0,:]  # cls token
            feats.append(feat.cpu())
            labels.append(labs)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels

def train_linear_probe(ckpt, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # dataloaders for train and test (full images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    train_ds = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
    test_ds = datasets.CIFAR10(root='data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MAE(img_size=32, patch_size=args.patch_size, embed_dim=args.embed_dim, encoder_depth=args.depth, mask_ratio=args.mask_ratio)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model = model.to(device)

    train_feats, train_labels = extract_features(model, train_loader, device)
    test_feats, test_labels = extract_features(model, test_loader, device)

    # simple linear classifier
    clf = nn.Linear(train_feats.size(1), 10)
    clf = clf.to(device)
    opt = SGD(clf.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        clf.train()
        perm = torch.randperm(train_feats.size(0))
        for i in range(0, train_feats.size(0), args.batch_size):
            idx = perm[i:i+args.batch_size]
            xb = train_feats[idx].to(device)
            yb = train_labels[idx].to(device)
            logits = clf(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # evaluate
        clf.eval()
        with torch.no_grad():
            logits = clf(test_feats.to(device))
            preds = logits.argmax(dim=1).cpu()
            acc = (preds == test_labels).float().mean().item()
        print(f"Epoch {epoch} linear probe test acc: {acc:.4f}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=6)
    args = parser.parse_args()
    train_linear_probe(args.ckpt, args)
