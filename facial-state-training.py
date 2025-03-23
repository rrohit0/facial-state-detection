import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def create_model(input_shape=(24, 24, 1), num_classes=4):
    """
    Create a CNN model for facial state classification.
    
    Args:
        input_shape: Image dimensions and channels
        num_classes: Number of target classes (4 for open, closed, yawn, no_yawn)
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Prevent overfitting
        Dense(num_classes, activation='softmax')  # 4 classes: open, closed, yawn, no_yawn
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_generators(data_dir, img_size=(24, 24), batch_size=32, validation_split=0.2):
    """
    Create train and validation data generators from image directory.
    
    Args:
        data_dir: Path to data directory containing class subdirectories
        img_size: Target image dimensions
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
    
    Returns:
        train_generator, validation_generator, class_indices
    """
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split
    )
    
    # Only rescaling for validation set
    val_datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=validation_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator, train_generator.class_indices

def train_model(model, train_generator, validation_generator, epochs=50, early_stopping=True):
    """
    Train the model with the given data generators.
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Maximum number of training epochs
        early_stopping: Whether to use early stopping
    
    Returns:
        Training history and trained model
    """
    # Callbacks
    callbacks = []
    
    # Model checkpoint to save best model
    model_checkpoint = ModelCheckpoint(
        'best_facial_state_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(model_checkpoint)
    
    # Early stopping to prevent overfitting
    if early_stopping:
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    return history, model

def evaluate_model(model, validation_generator, class_indices):
    """
    Evaluate the trained model and visualize results.
    
    Args:
        model: Trained Keras model
        validation_generator: Validation data generator
        class_indices: Dictionary mapping class names to indices
    """
    # Get class names from indices
    class_names = {v: k for k, v in class_indices.items()}
    
    # Predict on validation data
    validation_generator.reset()
    y_pred = []
    y_true = []
    
    batch_count = 0
    max_batches = len(validation_generator)
    
    # Collect predictions and true labels
    for X_batch, y_batch in validation_generator:
        y_pred_batch = model.predict(X_batch)
        y_pred.extend(np.argmax(y_pred_batch, axis=1))
        y_true.extend(np.argmax(y_batch, axis=1))
        
        batch_count += 1
        if batch_count >= max_batches:
            break
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(class_indices.keys())))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(class_indices.keys()),
                yticklabels=list(class_indices.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return y_true, y_pred

def save_model_summary(model, file_path='model_summary.txt'):
    """Save model architecture summary to a file with UTF-8 encoding."""
    with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved to {file_path}")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = r"D:\facial_state\data"  # Directory containing closed, open, yawn, no_yawn subdirectories
    IMG_SIZE = (24, 24)
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Prepare data
    print("Preparing data generators...")
    train_generator, validation_generator, class_indices = prepare_data_generators(
        DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    
    # Print class mapping
    print("\nClass Indices:")
    for class_name, idx in class_indices.items():
        print(f"{idx}: {class_name}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=len(class_indices))
    model.summary()
    save_model_summary(model)
    
    # Train model
    print("\nTraining model...")
    history, model = train_model(model, train_generator, validation_generator, epochs=EPOCHS)
    
    # Save final model
    model.save('facial_state_model.h5')
    print("\nModel saved as 'facial_state_model.h5'")
    
    # Evaluate model
    print("\nEvaluating model...")
    y_true, y_pred = evaluate_model(model, validation_generator, class_indices)
    
    print("\nTraining complete! The model can now be used for drowsiness detection.")
