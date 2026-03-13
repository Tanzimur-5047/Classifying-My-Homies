import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import os

IMG_SIZE = 224
BATCH_SIZE = 16
MODEL_PATH = 'model/best_model_ft.keras'
VAL_DIR = 'data/val'

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(val_generator.class_indices.keys())
print(f"Classes: {class_names}")

predictions = model.predict(val_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes

print("\n--- Classification Report ---")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

f1 = f1_score(true_classes, predicted_classes, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")

f1_per_class = f1_score(true_classes, predicted_classes, average=None)
for name, score in zip(class_names, f1_per_class):
    print(f"  F1 {name}: {score:.4f}")

cm = confusion_matrix(true_classes, predicted_classes)
print("\n--- Confusion Matrix ---")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png', dpi=150)
print("\nConfusion matrix saved to model/confusion_matrix.png")

print("\n--- Per-Class Accuracy ---")
for i, name in enumerate(class_names):
    class_total = cm[i].sum()
    class_correct = cm[i][i]
    accuracy = class_correct / class_total * 100
    print(f"  {name}: {class_correct}/{class_total} = {accuracy:.1f}%")