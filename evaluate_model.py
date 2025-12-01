import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# ====================================================================
# STEP 1: Configuration and Model/Data Loading
# ====================================================================

# --- Configuration (MUST match your training script) ---
# Ensure these paths and sizes are correct based on your setup
DATA_DIR = 'dataset'                  # Your dataset folder
MODEL_PATH = 'models/final_model.keras' # Path to your saved model
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
# Note: Assuming your split used a 'validation' subset for testing

# Create a simple generator for evaluation (NO data augmentation, only normalization)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) 

# Load the Test Data Generator
test_generator = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation', # This generator pulls the data used for testing/validation
    shuffle=False,       # IMPORTANT: Do not shuffle for accurate evaluation metrics
    seed=42
)

# Load the saved model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"\nModel loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"\nERROR: Could not load model from {MODEL_PATH}. Check the path.")
    print(e)
    exit()

# ====================================================================
# STEP 2: Generate Predictions and True Labels
# ====================================================================

# Calculate steps needed to process the whole dataset
steps_per_epoch = test_generator.samples // test_generator.batch_size
if test_generator.samples % test_generator.batch_size != 0:
    steps_per_epoch += 1

print("Generating predictions on the test set...")
# Make predictions
predictions = model.predict(test_generator, steps=steps_per_epoch)

# Convert predictions (probabilities) to class indices
y_pred_indices = np.argmax(predictions, axis=1)

# Get the true class indices and ensure array lengths match
y_true_indices = test_generator.classes[:len(y_pred_indices)] 

# Get the class names for labeling reports
class_names = list(test_generator.class_indices.keys())
print("Prediction generation complete.")

# ====================================================================
# STEP 3 & 4: Display Classification Report and Confusion Matrix
# ====================================================================

# --- Classification Report ---
print("\n" + "="*70)
print("ADVANCED MODEL EVALUATION REPORT")
print("="*70)
print(classification_report(y_true_indices, y_pred_indices, target_names=class_names))
print("="*70 + "\n")

# --- Confusion Matrix ---
cm = confusion_matrix(y_true_indices, y_pred_indices)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Confusion Matrix for Fruit Classification (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("Confusion Matrix displayed. Review cells outside the diagonal to find misclassifications.")


# ====================================================================
# STEP 5: Deployment Preparation (Reusable Function)
# ====================================================================

def classify_uploaded_fruit(img_path, loaded_model, class_labels):
    """
    Loads an image and uses the model to classify the fruit.
    """
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = loaded_model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_index]
        confidence = predictions[predicted_index] * 100
        
        return predicted_class, confidence

    except Exception as e:
        return f"An error occurred: {e}", 0

# --- Example Usage ---
# Use a path to a fruit image that the model hasn't seen before
# example_img_path = 'path/to/a/new/apple.jpg' 
# print("\n" + "="*70)
# print(f"DEMO CLASSIFICATION ON A SINGLE IMAGE:")
# class_result, conf = classify_uploaded_fruit(example_img_path, model, class_names)
# print(f"Predicted: {class_result} | Confidence: {conf:.2f}%")
# print("="*70)

print("\nEvaluation complete. Your next step is **Deployment** (e.g., building a web app).")