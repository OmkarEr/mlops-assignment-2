import os
import shutil
import mlflow
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        layers.InputLayer(shape=(224, 224, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prepare_dataset(source_dir, dest_dir):
    """The absolute ultimate filter: Uses TensorFlow's OWN decoder to verify every image."""
    
    if os.path.exists(dest_dir):
        print(f"Removing old {dest_dir} to apply TensorFlow-native filtering...")
        shutil.rmtree(dest_dir)

    print("Cleaning and organizing dataset (this will take ~30-60 seconds)...")
    os.makedirs(os.path.join(dest_dir, 'cats'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'dogs'), exist_ok=True)

    valid_count = 0
    corrupt_count = 0

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)

                try:
                    # THE FIX: If TensorFlow's C++ backend can't decode it, we don't want it.
                    img_bytes = tf.io.read_file(file_path)
                    _ = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
                except Exception:
                    corrupt_count += 1
                    continue

                # Assign to proper class
                name_lower = file.lower()
                folder_lower = os.path.basename(root).lower()

                if 'cat' in name_lower or 'cat' in folder_lower:
                    label = 'cats'
                elif 'dog' in name_lower or 'dog' in folder_lower:
                    label = 'dogs'
                else:
                    continue 

                # Create symlink
                dest_path = os.path.join(dest_dir, label, f"{valid_count}_{file}")
                os.symlink(os.path.abspath(file_path), os.path.abspath(dest_path))
                valid_count += 1

    print(f"Prepared {valid_count} valid TF-compatible images. Skipped {corrupt_count} bad ones.")
    return dest_dir

if __name__ == "__main__":
    mlflow.set_experiment("cats_vs_dogs")
    
    epochs = 3
    batch_size = 32
    
    prepare_dataset("./data", "./clean_data")
    
    print("Loading datasets for training...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "./clean_data",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "./clean_data",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=batch_size
    )

    with mlflow.start_run():
        print("Building baseline CNN...")
        model = build_model()
        
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        
        print("Training on real dataset...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        
        mlflow.log_metric("accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
            
        model.save("model.h5")
        mlflow.log_artifact("model.h5")
        print("Model built, trained, and tracked via MLflow.")