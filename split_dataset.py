import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        source_dir: Path to original dataset (should contain folders for each fruit class)
        output_dir: Path where split dataset will be saved
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
    """
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Get all class folders
    classes = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Process each class
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        
        # Get all images in this class
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        print(f"\nProcessing {class_name}: {len(images)} images")
        
        # Split into train and temp (val+test)
        train_imgs, temp_imgs = train_test_split(
            images, 
            test_size=(1 - train_ratio), 
            random_state=42
        )
        
        # Split temp into val and test
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=42
        )
        
        # Create class directories in each split
        for split in splits:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
        
        # Copy images to respective directories
        for img in train_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_dir, 'train', class_name, img)
            shutil.copy2(src, dst)
        
        for img in val_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_dir, 'val', class_name, img)
            shutil.copy2(src, dst)
        
        for img in test_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_dir, 'test', class_name, img)
            shutil.copy2(src, dst)
        
        print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
    
    print("\n✅ Dataset split completed!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # Configuration - Updated for your project structure
    SOURCE_DIR = "dataset/original"  # Your fruit folders location
    OUTPUT_DIR = "dataset/split"  # This will create train/val/test folders
    
    # Split dataset (70% train, 15% val, 15% test)
    split_dataset(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )