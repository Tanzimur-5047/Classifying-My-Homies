import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


with open('model/history.json', 'r') as f:
    history = json.load(f)

epochs = range(1, len(history['accuracy']) + 1)
phase_split = history['phase_split']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, history['accuracy'], label='Train Accuracy')
ax1.plot(epochs, history['val_accuracy'], label='Val Accuracy')
ax1.axvline(x=phase_split, color='gray', linestyle='--', label='Fine-tuning starts')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(epochs, history['loss'], label='Train Loss')
ax2.plot(epochs, history['val_loss'], label='Val Loss')
ax2.axvline(x=phase_split, color='gray', linestyle='--', label='Fine-tuning starts')
ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig('model/training_plot.png', dpi=150)
print("Plot saved to model/training_plot.png")