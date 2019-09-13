import numpy as np
import matplotlib.pyplot as plt

Loss=[0]
Val_Loss=[0]
Acc=[0]
Val_Acc=[0]

N = 1 # change this number according to your epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), Loss, label="train_loss")
plt.plot(np.arange(0, N), Val_Loss, label="val_loss")
plt.plot(np.arange(0, N), Acc, label="train_acc")
plt.plot(np.arange(0, N), Val_Acc, label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
