import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

print("=" * 60)
print("🚀 FRUIT CLASSIFICATION - ENHANCED TRAINING V2")
print("=" * 60)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

# Auto-detect correct path
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up from src/ to project root
DATA_DIR = os.path.join(project_root, 'dataset', 'split')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

print(f"\n📁 Dataset directory: {os.path.abspath(DATA_DIR)}")

# Check dataset exists
if not os.path.exists(DATA_DIR):
    print(f"❌ Error: Dataset directory not found")
    exit(1)

# ⭐ NEW: Check class distribution
print("\n📊 Analyzing class distribution...")
train_dir = os.path.join(DATA_DIR, 'train')
class_counts = {}
for fruit_class in os.listdir(train_dir):
    class_path = os.path.join(train_dir, fruit_class)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[fruit_class] = count
        print(f"   {fruit_class}: {count} images")

# Calculate class weights for imbalanced data
total_samples = sum(class_counts.values())
class_weight = {i: total_samples / (len(class_counts) * count) 
                for i, (cls, count) in enumerate(class_counts.items())}
print(f"\n⚖️  Class weights calculated for balanced training")

print("\n🎨 Setting up ULTRA-AGGRESSIVE data augmentation for small dataset...")
# ⭐ ULTRA-AGGRESSIVE: Essential for small datasets (27 images/class)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,           # Even more rotation
    width_shift_range=0.4,       # More shifting
    height_shift_range=0.4,      
    horizontal_flip=True,
    vertical_flip=True,          
    zoom_range=[0.7, 1.4],       # Both zoom in AND out
    shear_range=0.4,             
    brightness_range=[0.3, 1.8], # Extreme brightness variation
    channel_shift_range=50.0,    # Strong color shifts (helps separate red fruits)
    fill_mode='reflect'          # Better edge handling
)

val_datagen = ImageDataGenerator(rescale=1./255)

print("📊 Loading training data...")
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("📊 Loading validation data...")
validation_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes
print(f"\n✅ Data loaded successfully!")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {validation_generator.samples}")
print(f"   Number of classes: {num_classes}")
print(f"   Class names: {list(train_generator.class_indices.keys())}")

print("\n🏗️  Building ENHANCED model with MobileNetV2...")

# Using MobileNetV2 (reliable and works with all TF versions)
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model initially
base_model.trainable = False

# ⭐ ENHANCED: Deeper and more robust head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),                          # Increased dropout
    layers.Dense(512, activation='relu'),         # Larger layer
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),         # Additional layer
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# ⭐ IMPROVEMENT: Use label smoothing to reduce overconfidence
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

print("✅ Model compiled successfully!")
print(f"\n📋 Model Summary:")
model.summary()

# Enhanced callbacks
checkpoint = ModelCheckpoint(
    'models/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=12,              # Increased patience
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stop, reduce_lr]

print("\n" + "=" * 60)
print("🎯 Starting Training Phase 1: Transfer Learning")
print("=" * 60)

# Phase 1: Train with frozen base model and class weights
history1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,                    # Increased from 20
    callbacks=callbacks,
    class_weight=class_weight,    # ⭐ NEW: Handle class imbalance
    verbose=1
)

print("\n" + "=" * 60)
print("🔓 Starting Training Phase 2: Fine-tuning")
print("=" * 60)

# Phase 2: Unfreeze and fine-tune
base_model.trainable = True

# Freeze first 80% of layers
num_layers = len(base_model.layers)
for layer in base_model.layers[:int(num_layers * 0.8)]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00005),  # Even lower LR
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

# Continue training
history2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS - 25,
    initial_epoch=len(history1.history['loss']),
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# Combine histories
history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

# Save final model
print("\n💾 Saving final model...")
model.save('models/final_model_v2.keras')
print("✅ Model saved as: models/final_model_v2.keras")

# Plot training history
print("\n📊 Creating training plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logs/training_history_v2.png', dpi=150, bbox_inches='tight')
print("✅ Training plots saved as: logs/training_history_v2.png")
plt.show()

# Print final results
print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📈 Final Results:")
print(f"   Training Accuracy: {history['accuracy'][-1]:.4f}")
print(f"   Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
print(f"   Training Loss: {history['loss'][-1]:.4f}")
print(f"   Validation Loss: {history['val_loss'][-1]:.4f}")

print(f"\n💾 Model saved: models/final_model_v2.keras")
print(f"\n🎯 Next step: Test your model!")
print("=" * 60)