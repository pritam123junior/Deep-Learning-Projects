from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=40, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Load testing data
validation_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

if __name__ == "__main__":
    print("Data generators created successfully")
