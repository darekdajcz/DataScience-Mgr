#!/usr/bin/env python3
# boat_cnn_cpu.py — full pipeline CPU-only, bez Inductora
###############################################################################

import os, json, time, platform, torch
from pathlib import Path
from collections import Counter
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# ───────────── DISABLE TORCH.INDUCTOR / DYNAMO (Windows bez MSVC) ────────────
os.environ["TORCHINDUCTOR_DISABLE"] = "1"     # <-- blokuje próby kompilacji C++
import torch._dynamo as dynamo
dynamo.disable()                              # <-- całkiem wyłącza Dynamo

# ───────────── PARAMETRY ─────────────────────────────────────────────────────
DATA_PATH   = r"C:/Users/dariu/Desktop/PJATK/boat_category/boat_category"
IMG_SIZE    = 256         # 256 przy mocnym CPU
BATCH_SIZE  = 128
VAL_SPLIT   = 0.15         # val = test = 15 %
EPOCHS      = 100
LR          = 3e-4
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔧  PyTorch {torch.__version__} • Python {platform.python_version()} • Device = {DEVICE}")

# ───────────── DANE ──────────────────────────────────────────────────────────
tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
tf_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

full_ds   = datasets.ImageFolder(Path(DATA_PATH), transform=tf_train)
class_map = full_ds.classes
n_total   = len(full_ds)
print(f"📂  Obrazów: {n_total}  |  Klasy: {class_map}")
for idx, cls in enumerate(class_map):
    print(f"   • {cls:<12}: {Counter(l for _, l in full_ds.samples)[idx]}")

n_val   = int(n_total * VAL_SPLIT)
n_test  = n_val
n_train = n_total - n_val - n_test
print(f"🔀  Podział → train={n_train}, val={n_val}, test={n_test}")

g = torch.Generator().manual_seed(SEED)
train_ds, val_ds, test_ds = random_split(full_ds,
                                         [n_train, n_val, n_test],
                                         generator=g)
val_ds.dataset.transform  = tf_eval
test_ds.dataset.transform = tf_eval

dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True,  num_workers=0, pin_memory=False)
dl_val   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0, pin_memory=False)
dl_test  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0, pin_memory=False)

# ───────────── MODEL ─────────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self, n_cls: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, n_cls),
        )
    def forward(self, x): return self.net(x)

model     = TinyCNN(len(class_map)).to(DEVICE)   # <-- BEZ torch.compile
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ───────────── PĘTLA TRENINGOWA ──────────────────────────────────────────────
history = {"train_loss":[], "val_loss":[], "test_loss":[],
           "train_acc":[],  "val_acc":[],  "test_acc":[]}

def run_epoch(loader, train=False):
    model.train(train)
    loss_sum = correct = total = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        if train: optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        if train:
            loss.backward(); optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum/total, correct/total

print("\n🚀  Start treningu…\n")
for ep in range(1, EPOCHS+1):
    t0 = time.time()
    tl, ta = run_epoch(dl_train, train=True)
    vl, va = run_epoch(dl_val,   train=False)
    te, ta_test = run_epoch(dl_test, train=False)      # ← DODANE
    dt = time.time() - t0

    history["train_loss"].append(tl);   history["train_acc"].append(ta)
    history["val_loss"].append(vl);     history["val_acc"].append(va)
    history["test_loss"].append(te);    history["test_acc"].append(ta_test)

    print(f"[{ep:02}] tr {ta:.3f}/{tl:.4f} | "
          f"val {va:.3f}/{vl:.4f} | "
          f"test {ta_test:.3f}/{te:.4f}  ({dt:.1f}s)")

# ───────────── EWALUACJA ──────────────────────────────────────────────────────
print("\n📝  Test set…")
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for X, y in dl_test:
        logits = model(X.to(DEVICE))
        y_true.extend(y.numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())
print(classification_report(y_true, y_pred, target_names=class_map))

# ───────────── ZAPIS HISTORII + WYKRESY ──────────────────────────────────────
with open("history.json", "w") as f: json.dump(history, f, indent=2)

epochs = range(1, EPOCHS+1)
def plot_curve(y_tr, y_val, title, ylabel, fname):
    plt.figure(figsize=(6,4))
    plt.plot(epochs, y_tr, label="train")
    plt.plot(epochs, y_val, label="val")
    plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(fname); plt.close()

plot_curve(history["train_acc"], history["val_acc"],
           "Learning curve – accuracy", "Accuracy", "acc_curve.png")
plot_curve(history["train_loss"], history["val_loss"],
           "Learning curve – loss", "Loss", "loss_curve.png")

print("\n✅  Zapisano: history.json, acc_curve.png, loss_curve.png")
###############################################################################
