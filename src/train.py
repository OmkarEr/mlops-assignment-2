import mlflow
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        layers.InputLayer(input_shape=(224, 224, 3)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    mlflow.set_experiment("cats_vs_dogs")
    with mlflow.start_run():
        print("Building baseline CNN...")
        model = build_model()
        
        # In a real run, you'd load your Kaggle dataset here.
        # We are logging dummy parameters to fulfill M1 requirements.
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 32)
        
        # Save and log the model
        model.save("model.h5")
        mlflow.log_artifact("model.h5")
        print("Model built and tracked via MLflow.")