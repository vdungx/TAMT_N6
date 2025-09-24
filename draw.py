import torch
import matplotlib.pyplot as plt

# Đường dẫn đến file trlog của bạn
trlog_path = r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\checkpoints\SSv2Full\VideoMAES_meta_deepbdc_5way_5shot_2TAA\trlog"

# Load dữ liệu
trlog = torch.load(trlog_path)



train_acc = trlog['train_acc']
val_acc = trlog['val_acc']
train_loss = trlog['train_loss']
val_loss = trlog['val_loss']

print(train_acc)
print(val_acc)
print(train_loss)
print(val_loss)

epochs = range(len(train_acc))

# Vẽ Accuracy
plt.figure(figsize=(10,5))
plt.plot(epochs, train_acc, label="Training Accuracy", color='red', linestyle='--')
plt.plot(epochs, val_acc, label="Validation Accuracy", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy Curve")
plt.legend()
plt.grid(True)
plt.show()

# Vẽ Loss
plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss, label="Training Loss", color='blue', linestyle='--')
plt.plot(epochs, val_loss, label="Validation Loss", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.show()
