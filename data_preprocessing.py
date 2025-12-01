print("Starting script...")

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install required packages:")
    print("pip install tensorflow matplotlib numpy pillow")
    exit(1)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32

# Smart path detection
def find_dataset_path():
    """Automatically find the dataset directory"""
    print("\n🔍 Detecting dataset path...")
    print(f"Current directory: {os.getcwd()}")
    
    possible_paths = [
        'dataset/split',                                    # If running from project root
        'fruit_classification/dataset/split',               # If running from parent dir
        '../dataset/split',                                 # If running from src folder
        os.path.join(os.path.dirname(__file__), '..', 'dataset', 'split'),  # Relative to script
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        print(f"   Checking: {abs_path}")
        if os.path.exists(path):
            print(f"   ✅ Found at: {abs_path}")
            return path
    
    # If not found, ask user
    print("\n⚠️  Could not auto-detect dataset path!")
    print("\nPlease check your folder structure:")
    print("It should be: dataset/split/train/, dataset/split/val/, dataset/split/test/")
    custom_path = input("\nEnter the path to your dataset/split folder: ").strip().strip('"').strip("'")
    return custom_path

DATA_DIR = find_dataset_path()

def create_data_generators():
    """
    Create data generators with IMPROVED augmentation
    including brightness_range to handle light and dark colored fruits
    """
    print(f"\n🔍 Using dataset path: {os.path.abspath(DATA_DIR)}")
    
    # Check if directory exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {os.path.abspath(DATA_DIR)}")
    
    # Check for train/val/test folders
    required_folders = ['train', 'val', 'test']
    for folder in required_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"\n❌ Missing required folder: {folder}")
            print(f"   Looking for: {os.path.abspath(folder_path)}")
            raise FileNotFoundError(f"Required folder not found: {folder_path}")
    
    # IMPROVED: Training data generator with brightness augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        brightness_range=[0.5, 1.5],  # ⭐ KEY FIX: Handle light & dark variations
        fill_mode='nearest'
    )
    
    # Validation and test data generators (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    print("📂 Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    # Load validation data
    print("📂 Loading validation data...")
    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Load test data
    print("📂 Loading test data...")
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def visualize_augmentation(generator, num_images=5):
    """
    Visualize augmented images including brightness variations
    """
    print("\n🎨 Generating augmented image samples...")
    images, labels = next(generator)
    
    plt.figure(figsize=(15, 3))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        class_idx = np.argmax(labels[i])
        class_name = list(generator.class_indices.keys())[class_idx]
        plt.title(f'{class_name}')
    
    plt.tight_layout()
    plt.savefig('augmented_samples.png')
    print("💾 Augmented samples saved as 'augmented_samples.png'")
    plt.close()

# Main execution
print("\n" + "="*50)
print("🚀 FRUIT CLASSIFICATION - DATA PREPROCESSING")
print("="*50)

try:
    print("\n🔄 Creating data generators...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training samples: {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Test samples: {test_gen.samples}")
    print(f"   Number of classes: {train_gen.num_classes}")
    print(f"   Class names: {list(train_gen.class_indices.keys())}")
    
    visualize_augmentation(train_gen)
    
    print("\n✅ Data preprocessing setup complete!")
    print("💡 Key improvement: brightness_range=[0.5, 1.5] added")
    print("   This will help model recognize both light & dark colored fruits!")
    print("\n📝 Next step: Run train_improved.py to retrain your model")
    print("="*50)
    
except FileNotFoundError as e:
    print(f"\n❌ Error: {e}")
    print(f"\n💡 Troubleshooting tips:")
    print(f"   1. Make sure you're in the fruit_classification directory")
    print(f"   2. Check your dataset folder structure:")
    print(f"      dataset/split/train/apple/, dataset/split/train/banana/, etc.")
    print(f"      dataset/split/val/apple/, dataset/split/val/banana/, etc.")
    print(f"      dataset/split/test/apple/, dataset/split/test/banana/, etc.")
    print(f"   3. Verify with: dir dataset\\split")
except Exception as e:
    print(f"\n❌ Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()