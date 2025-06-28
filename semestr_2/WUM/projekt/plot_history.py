import json, matplotlib.pyplot as plt, os

with open("history.json") as f:
    hist = json.load(f)

epochs = list(range(1, len(hist["train_loss"]) + 1))
train_loss, val_loss = hist["train_loss"], hist["val_loss"]
train_acc,  val_acc  = hist["train_acc"],  hist["val_acc"]

# heurystycznie: pierwsza epoka, od której val_loss rośnie 2× z rzędu
min_ep = val_loss.index(min(val_loss))
over = None
for i in range(min_ep+1, len(val_loss)-2):
    if val_loss[i] < val_loss[i+1] < val_loss[i+2]:
        over = i+2   # 1-indeks
        break

def plot_curve(y_tr, y_val, ylabel, title, fname):
    plt.figure(figsize=(7,4))
    plt.plot(epochs, y_tr, label="train")
    plt.plot(epochs, y_val, label="val")
    if over:
        plt.axvline(over, color="gray", ls="--")
        plt.text(over+0.2, max(max(y_tr), max(y_val))*0.9,
                 f"overfit start (ep {over})", rotation=90, va="top")
    plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(fname); plt.close()

plot_curve(train_acc, val_acc, "Accuracy", "Accuracy curve", "acc_curve.png")
plot_curve(train_loss, val_loss, "Loss", "Loss curve", "loss_curve.png")

print("✔️  acc_curve.png & loss_curve.png zapisane z zaznaczonym punktem overfittingu.")
