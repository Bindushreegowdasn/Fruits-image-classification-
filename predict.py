import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def predict_fruit(model, image_path, class_names):
    """Predict the fruit in an image"""
    img_array, original_img = load_and_preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    predicted_fruit = class_names[predicted_class_idx]
    
    return predicted_fruit, confidence, predictions[0], original_img

def visualize_prediction(image_path, predicted_fruit, confidence, all_predictions, class_names):
    """Display the image with prediction results"""
    _, img = load_and_preprocess_image(image_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show image
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'Predicted: {predicted_fruit}\nConfidence: {confidence:.2f}%', 
                  fontsize=16, fontweight='bold', color='green')
    
    # Show prediction probabilities
    colors = ['#4CAF50' if i == np.argmax(all_predictions) else '#CCCCCC' 
              for i in range(len(class_names))]
    ax2.barh(class_names, all_predictions * 100, color=colors)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('All Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    
    # Add percentage labels on bars
    for i, v in enumerate(all_predictions * 100):
        ax2.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save result
    os.makedirs('logs', exist_ok=True)
    plt.savefig('logs/prediction_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Prediction visualization saved to: logs/prediction_result.png")

def select_image():
    """Open file browser to select an image"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring to front
    
    print("\n📁 Opening file browser... (check if a window popped up)")
    
    file_path = filedialog.askopenfilename(
        title="Select a fruit image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def main():
    print("=" * 60)
    print("🍎 FRUIT CLASSIFIER - EASY PREDICTION TOOL 🍌")
    print("=" * 60)
    
    # Load the trained model
    model_path = "models/final_model.keras"
    print(f"\n📦 Loading model from: {model_path}")
    
    try:
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Define your fruit classes (based on your screenshot)
    class_names = [
        'apple', 'banana', 'cherry', 'chickoo', 'grapes', 
        'kiwi', 'mango', 'orange', 'strawberry'
    ]
    
    print(f"\n🏷️  Fruit Classes: {', '.join(class_names)}")
    
    # Method selection
    print("\n" + "-" * 60)
    print("Choose how to select your image:")
    print("1. Browse and select (recommended)")
    print("2. Type the path manually")
    print("-" * 60)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        image_path = select_image()
        if not image_path:
            print("\n❌ No image selected. Exiting.")
            return
    else:
        image_path = input("\n📸 Enter image path: ").strip()
        # Remove quotes if user copied path with quotes
        image_path = image_path.strip('"').strip("'")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"\n❌ File not found: {image_path}")
        return
    
    print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
    
    # Make prediction
    try:
        predicted_fruit, confidence, all_predictions, _ = predict_fruit(
            model, image_path, class_names
        )
        
        print("\n" + "=" * 60)
        print("📊 PREDICTION RESULTS")
        print("=" * 60)
        print(f"\n🎯 Predicted Fruit: {predicted_fruit.upper()}")
        print(f"💯 Confidence: {confidence:.2f}%")
        
        print(f"\n📈 All Probabilities:")
        for fruit, prob in zip(class_names, all_predictions):
            bar = "█" * int(prob * 40)
            print(f"  {fruit:12s}: {prob*100:6.2f}% {bar}")
        
        # Visualize results
        print(f"\n📊 Creating visualization...")
        visualize_prediction(image_path, predicted_fruit, confidence, 
                           all_predictions, class_names)
        
        print("\n" + "=" * 60)
        print("✨ Done! Check the visualization window.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()