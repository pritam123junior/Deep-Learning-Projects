import tensorflow as tf
from data_preparation import validation_generator

# Load the model
model = tf.keras.models.load_model('../models/leaf_disease_detector.h5')

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {accuracy*100:.2f}%')

if __name__ == "__main__":
    print("Model evaluated successfully")
