import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("🚀 FRUIT CLASSIFICATION - IMPROVED TRAINING")
print("=" * 60)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
DATA_DIR = 'dataset/split'  # ✅ FIXED: Correct path to split folder

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

print(f"\n📁 Dataset directory: {os.path.abspath(DATA_DIR)}")

# Check if dataset exists
if not os.path.exists(DATA_DIR):
    print(f"❌ Error: Dataset directory not found at {os.path.abspath(DATA_DIR)}")
    print("Please make sure you have dataset/split/train, dataset/split/val, dataset/split/test")
    exit(1)

print("\n🎨 Setting up enhanced data augmentation...")
# Enhanced training data generator with BRIGHTNESS augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.5, 1.5],  # ⭐ KEY: Handles light & dark fruits
    fill_mode='nearest'
)

# Validation data generator (only rescaling)
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

print("\n🏗️  Building model with transfer learning (MobileNetV2)...")

# Load pre-trained MobileNetV2
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model initially
base_model.trainable = False

# Build model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled successfully!")
print(f"\n📋 Model Summary:")
model.summary()

# Callbacks
checkpoint = ModelCheckpoint(
    'models/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
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

# Phase 1: Train with frozen base model
history1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 60)
print("🔓 Starting Training Phase 2: Fine-tuning")
print("=" * 60)

# Phase 2: Unfreeze base model and fine-tune
base_model.trainable = True

# Freeze first 100 layers, fine-tune the rest
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS - 20,
    initial_epoch=len(history1.history['loss']),
    callbacks=callbacks,
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
model.save('models/final_model.keras')
print("✅ Model saved as: models/final_model.keras")

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
plt.savefig('logs/training_history.png', dpi=150, bbox_inches='tight')
print("✅ Training plots saved as: logs/training_history.png")
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

print(f"\n💾 Models saved:")
print(f"   Best model: models/best_model.keras")
print(f"   Final model: models/final_model.keras")

print(f"\n🎯 Next step: Test your model with predict.py!")
print(f"   python src/predict.py")
print("=" * 60)