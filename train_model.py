print("Starting training script...")

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Import from other scripts
from data_preprocessing import create_data_generators
from build_model import create_fruit_classifier

# Configuration
EPOCHS = 30
BATCH_SIZE = 32

def create_callbacks():
    """
    Create callbacks for training
    """
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # ModelCheckpoint - save best model
    checkpoint = ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # EarlyStopping - stop if no improvement
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # ReduceLROnPlateau - reduce learning rate when plateauing
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    return [checkpoint, early_stop, reduce_lr]

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    print("\n📊 Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'logs/training_history_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Training history saved: {plot_path}")
    plt.close()

def train_model():
    """
    Main training function
    """
    print("\n" + "="*60)
    print("🎯 FRUIT CLASSIFICATION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    print(f"\n📊 Training Configuration:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Training samples: {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Steps per epoch: {train_gen.samples // BATCH_SIZE}")
    print(f"   Validation steps: {val_gen.samples // BATCH_SIZE}")
    
    # Build model
    print("\n🏗️  Building model...")
    model = create_fruit_classifier()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "-" * 60)
    print("✅ Training completed!")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    
    print(f"\n📊 Final Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Save final model
    final_model_path = 'models/final_model.keras'
    model.save(final_model_path)
    print(f"\n💾 Final model saved: {final_model_path}")
    
    # Get best metrics
    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    print(f"\n🏆 Best Performance:")
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   Best Validation Loss: {best_val_loss:.4f}")
    
    print("\n" + "="*60)
    print("🎉 Training pipeline completed successfully!")
    print("="*60)
    
    return history, model

if __name__ == "__main__":
    try:
        history, model = train_model()
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()