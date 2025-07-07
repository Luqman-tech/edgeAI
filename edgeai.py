# Edge AI Recyclable Item Classification
# Complete implementation with TensorFlow Lite conversion and deployment

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import json
import time

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class RecyclableClassifier:
    def __init__(self, img_size=(224, 224), num_classes=4):
        """
        Initialize the recyclable item classifier
        
        Args:
            img_size: Input image dimensions
            num_classes: Number of recyclable categories
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = ['plastic', 'paper', 'glass', 'metal']
        self.model = None
        self.tflite_model = None
        
    def create_model(self):
        """Create a lightweight CNN model optimized for edge deployment"""
        
        # Using MobileNetV2 as base - optimized for mobile/edge devices
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            alpha=0.75,  # Width multiplier for efficiency
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers for faster training
        base_model.trainable = False
        
        # Add custom classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile with optimization for edge deployment
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def create_sample_dataset(self, samples_per_class=100):
        """
        Create a synthetic dataset for demonstration
        In practice, you'd load real recyclable item images
        """
        print("Creating sample dataset...")
        
        # Generate synthetic data with different patterns for each class
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        for class_idx in range(self.num_classes):
            # Training data
            for _ in range(samples_per_class):
                # Create synthetic images with class-specific patterns
                if class_idx == 0:  # Plastic - smooth textures
                    img = np.random.rand(*self.img_size, 3) * 0.3 + 0.4
                elif class_idx == 1:  # Paper - textured patterns
                    img = np.random.rand(*self.img_size, 3) * 0.8 + 0.1
                    img += np.random.normal(0, 0.1, img.shape)  # Add noise
                elif class_idx == 2:  # Glass - reflective patterns
                    img = np.random.rand(*self.img_size, 3) * 0.6 + 0.2
                    img[:, :, 2] += 0.2  # Blue tint
                else:  # Metal - metallic patterns
                    img = np.random.rand(*self.img_size, 3) * 0.4 + 0.5
                    img[:, :, 0] += 0.1  # Red tint
                
                img = np.clip(img, 0, 1)
                X_train.append(img)
                y_train.append(class_idx)
            
            # Test data (20% of training size)
            for _ in range(samples_per_class // 5):
                if class_idx == 0:
                    img = np.random.rand(*self.img_size, 3) * 0.3 + 0.4
                elif class_idx == 1:
                    img = np.random.rand(*self.img_size, 3) * 0.8 + 0.1
                    img += np.random.normal(0, 0.1, img.shape)
                elif class_idx == 2:
                    img = np.random.rand(*self.img_size, 3) * 0.6 + 0.2
                    img[:, :, 2] += 0.2
                else:
                    img = np.random.rand(*self.img_size, 3) * 0.4 + 0.5
                    img[:, :, 0] += 0.1
                
                img = np.clip(img, 0, 1)
                X_test.append(img)
                y_test.append(class_idx)
        
        # Convert to numpy arrays
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int32)
        
        print(f"Dataset created: {len(X_train)} training samples, {len(X_test)} test samples")
        return X_train, y_train, X_test, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=10):
        """Train the model with early stopping and validation"""
        
        print("Training model...")
        
        # Add data augmentation for better generalization
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        
        # Callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=3, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance and generate metrics"""
        
        print("\nEvaluating model...")
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        
        # Generate classification report
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Recyclable Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, report, cm
    
    def convert_to_tflite(self, quantize=True):
        """Convert model to TensorFlow Lite format"""
        
        print("Converting model to TensorFlow Lite...")
        
        # Initialize converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Apply quantization for smaller model size
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save the model
        with open('recyclable_classifier.tflite', 'wb') as f:
            f.write(tflite_model)
        
        self.tflite_model = tflite_model
        
        # Compare model sizes
        original_size = os.path.getsize('recyclable_classifier.tflite') / (1024 * 1024)
        print(f"TensorFlow Lite model size: {original_size:.2f} MB")
        
        return tflite_model
    
    def benchmark_inference(self, X_test, num_samples=100):
        """Benchmark inference speed for both models"""
        
        print("\nBenchmarking inference speed...")
        
        # Prepare test samples
        test_samples = X_test[:num_samples]
        
        # Benchmark original model
        start_time = time.time()
        _ = self.model.predict(test_samples)
        original_time = time.time() - start_time
        
        # Benchmark TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        start_time = time.time()
        for sample in test_samples:
            interpreter.set_tensor(input_details[0]['index'], 
                                 sample.reshape(1, *self.img_size, 3))
            interpreter.invoke()
        tflite_time = time.time() - start_time
        
        print(f"Original model: {original_time:.3f}s ({original_time/num_samples*1000:.1f}ms per image)")
        print(f"TensorFlow Lite: {tflite_time:.3f}s ({tflite_time/num_samples*1000:.1f}ms per image)")
        print(f"Speedup: {original_time/tflite_time:.2f}x")
        
        return original_time, tflite_time
    
    def test_tflite_inference(self, X_test, y_test, num_samples=50):
        """Test TensorFlow Lite model accuracy"""
        
        print("\nTesting TensorFlow Lite model...")
        
        # Initialize interpreter
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test on sample data
        correct = 0
        total = min(num_samples, len(X_test))
        
        for i in range(total):
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], 
                                 X_test[i].reshape(1, *self.img_size, 3))
            
            # Run inference
            interpreter.invoke()
            
            # Get prediction
            output = interpreter.get_tensor(output_details[0]['index'])
            predicted = np.argmax(output)
            
            if predicted == y_test[i]:
                correct += 1
        
        accuracy = correct / total
        print(f"TensorFlow Lite accuracy: {accuracy:.3f}")
        
        return accuracy

def generate_deployment_guide():
    """Generate deployment guide for Raspberry Pi"""
    
    deployment_guide = """
# Raspberry Pi Deployment Guide

## Hardware Requirements
- Raspberry Pi 4 (2GB RAM minimum)
- MicroSD card (16GB+)
- Camera module or USB webcam
- Power supply

## Software Setup

### 1. Install Dependencies
```bash
sudo apt update
sudo apt install python3-pip python3-venv
pip3 install tensorflow-lite-runtime pillow numpy opencv-python
```

### 2. Deploy Model
```python
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import cv2

class EdgeRecyclableClassifier:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.class_names = ['plastic', 'paper', 'glass', 'metal']
    
    def preprocess_image(self, image_path):
        img = Image.open(image_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        return img_array.reshape(1, 224, 224, 3).astype(np.float32)
    
    def predict(self, image_path):
        input_data = self.preprocess_image(image_path)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_class = np.argmax(output)
        confidence = np.max(output)
        return self.class_names[predicted_class], confidence

# Usage
classifier = EdgeRecyclableClassifier('recyclable_classifier.tflite')
result, confidence = classifier.predict('test_image.jpg')
print(f"Prediction: {result} (Confidence: {confidence:.2f})")
```

### 3. Real-time Camera Integration
```python
import cv2

def real_time_classification():
    cap = cv2.VideoCapture(0)
    classifier = EdgeRecyclableClassifier('recyclable_classifier.tflite')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame temporarily
        cv2.imwrite('temp_frame.jpg', frame)
        
        # Classify
        result, confidence = classifier.predict('temp_frame.jpg')
        
        # Display result
        cv2.putText(frame, f'{result}: {confidence:.2f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Recyclable Classifier', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

## Performance Optimization Tips
1. Use quantized models for faster inference
2. Implement frame skipping for real-time applications
3. Use threading for camera capture and inference
4. Cache preprocessing results when possible
"""
    
    with open('deployment_guide.md', 'w') as f:
        f.write(deployment_guide)
    
    print("Deployment guide saved to 'deployment_guide.md'")

def main():
    """Main execution function"""
    
    print("=== Edge AI Recyclable Classification Project ===\n")
    
    # Initialize classifier
    classifier = RecyclableClassifier()
    
    # Create and train model
    print("Step 1: Creating model...")
    model = classifier.create_model()
    print(f"Model created with {model.count_params():,} parameters")
    
    # Create sample dataset
    print("\nStep 2: Preparing dataset...")
    X_train, y_train, X_test, y_test = classifier.create_sample_dataset()
    
    # Train model
    print("\nStep 3: Training model...")
    history = classifier.train_model(X_train, y_train, X_test, y_test, epochs=5)
    
    # Evaluate model
    print("\nStep 4: Evaluating model...")
    accuracy, report, cm = classifier.evaluate_model(X_test, y_test)
    
    print(f"\nModel Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    for class_name in classifier.class_names:
        metrics = report[class_name]
        print(f"{class_name}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Convert to TensorFlow Lite
    print("\nStep 5: Converting to TensorFlow Lite...")
    tflite_model = classifier.convert_to_tflite()
    
    # Benchmark performance
    print("\nStep 6: Benchmarking performance...")
    original_time, tflite_time = classifier.benchmark_inference(X_test)
    
    # Test TensorFlow Lite accuracy
    tflite_accuracy = classifier.test_tflite_inference(X_test, y_test)
    
    # Generate deployment guide
    print("\nStep 7: Generating deployment guide...")
    generate_deployment_guide()
    
    # Summary report
    print("\n" + "="*50)
    print("PROJECT SUMMARY")
    print("="*50)
    print(f"Original Model Accuracy: {accuracy:.3f}")
    print(f"TensorFlow Lite Accuracy: {tflite_accuracy:.3f}")
    print(f"Model Size: {os.path.getsize('recyclable_classifier.tflite') / (1024*1024):.2f} MB")
    print(f"Inference Speed Improvement: {original_time/tflite_time:.2f}x")
    print(f"Average Inference Time: {tflite_time/100*1000:.1f}ms per image")
    
    print("\nFiles Generated:")
    print("- recyclable_classifier.tflite (Edge AI model)")
    print("- confusion_matrix.png (Performance visualization)")
    print("- deployment_guide.md (Raspberry Pi deployment instructions)")
    
    print("\nEdge AI Benefits Demonstrated:")
    print("✓ Reduced model size for resource-constrained devices")
    print("✓ Faster inference for real-time applications")
    print("✓ Offline operation capability")
    print("✓ Privacy-preserving local processing")
    print("✓ Reduced bandwidth requirements")

if __name__ == "__main__":
    main()