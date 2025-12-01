print("Starting model building script...")

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Configuration
IMG_SIZE = 224
NUM_CLASSES = 9  # Your 9 fruit classes
LEARNING_RATE = 0.0001

def build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    Build a model using MobileNetV2 with transfer learning
    """
    print("\n🏗️  Building model with MobileNetV2...")
    
    # Load pre-trained MobileNetV2 (trained on ImageNet)
    # include_top=False means we remove the final classification layer
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers (we'll use pre-trained features)
    base_model.trainable = False
    print(f"   Base model loaded: {base_model.name}")
    print(f"   Trainable: {base_model.trainable}")
    
    # Add custom classification layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Convert features to 1D
    x = Dense(512, activation='relu')(x)  # Fully connected layer
    x = Dropout(0.5)(x)  # Dropout for regularization
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # Output layer
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print(f"\n📊 Model Architecture:")
    print(f"   Input shape: ({img_size}, {img_size}, 3)")
    print(f"   Base model layers: {len(base_model.layers)}")
    print(f"   Total layers: {len(model.layers)}")
    print(f"   Output classes: {num_classes}")
    
    return model

def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile the model with optimizer, loss, and metrics
    """
    print(f"\n⚙️  Compiling model...")
    print(f"   Optimizer: Adam (lr={learning_rate})")
    print(f"   Loss: Categorical Crossentropy")
    print(f"   Metrics: Accuracy")
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_fruit_classifier():
    """
    Create and compile the complete fruit classification model
    """
    model = build_model()
    model = compile_model(model)
    return model

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🍎 FRUIT CLASSIFICATION - MODEL BUILDING")
    print("="*60)
    
    try:
        # Create the model
        model = create_fruit_classifier()
        
        # Display model summary
        print("\n📋 Model Summary:")
        print("-" * 60)
        model.summary()
        
        # Count trainable parameters
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        print("\n📊 Parameter Count:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Non-trainable parameters: {non_trainable_params:,}")
        
        # Save model architecture
        os.makedirs('models', exist_ok=True)
        
        print("\n✅ Model built successfully!")
        print(f"   Model is ready for training")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()