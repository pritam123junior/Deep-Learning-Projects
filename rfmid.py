
!mkdir RFMiD_project

# Commented out IPython magic to ensure Python compatibility.
# prompt: cd /content/LDD_project

# %cd /content/RFMiD_project/

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    MobileNetV2, NASNetMobile, DenseNet201, InceptionResNetV2, ResNet152V2,
    NASNetLarge, VGG19, InceptionV3, Xception
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def get_pretrained_model(model_name, input_shape, num_classes):
    base_model = None
    preprocess_input = None

    if model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = mobilenet_preprocess
    elif model_name == 'NASNetMobile':
        base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = nasnet_preprocess
    elif model_name == 'DenseNet201':
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = densenet_preprocess
    elif model_name == 'InceptionResNetV2':
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = inception_resnet_preprocess
    elif model_name == 'ResNet152V2':
        base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = resnet_preprocess
    elif model_name == 'NASNetLarge':
        base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = nasnet_preprocess
    elif model_name == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = vgg_preprocess
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = inception_preprocess
    elif model_name == 'Xception':
        base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = xception_preprocess
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, preprocess_input



import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    MobileNetV2, NASNetMobile, DenseNet201, InceptionResNetV2, ResNet152V2,
    NASNetLarge, VGG19, InceptionV3, Xception
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Paths
train_dir = 'dataset/train'
validation_dir = 'dataset/test'

# List all image files
train_files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
validation_files = [f for f in os.listdir(validation_dir) if os.path.isfile(os.path.join(validation_dir, f))]

# Assuming binary classification with random labels
num_classes = 2

# Create DataFrame with filenames and labels
train_df = pd.DataFrame({
    'filename': train_files,
    'label': np.random.randint(0, num_classes, size=len(train_files))  # Random labels for illustration
})

validation_df = pd.DataFrame({
    'filename': validation_files,
    'label': np.random.randint(0, num_classes, size=len(validation_files))
})

# Image size and batch size
image_size = (224, 224)
batch_size = 32
# Create DataFrame with filenames and labels
train_df = pd.DataFrame({
    'filename': train_files,
    'label': ['class_' + str(np.random.randint(0, num_classes)) for _ in range(len(train_files))]  # String labels
})

validation_df = pd.DataFrame({
    'filename': validation_files,
    'label': ['class_' + str(np.random.randint(0, num_classes)) for _ in range(len(validation_files))]
})

# Create data generators
def create_datagen(preprocess_input):
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# Create datasets
def create_dataset(generator, dataframe, directory, image_size):
    return generator.flow_from_dataframe(
        dataframe,
        directory=directory,
        x_col='filename',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

# Create data generators with preprocessing
train_generator = create_datagen(mobilenet_preprocess)
validation_generator = create_datagen(mobilenet_preprocess)

train_dataset = create_dataset(train_generator, train_df, train_dir, image_size)
validation_dataset = create_dataset(validation_generator, validation_df, validation_dir, image_size)

# Define the models to train
model_names = [
    'MobileNetV2', 'NASNetMobile', 'DenseNet201', 'InceptionResNetV2',
    'ResNet152V2', 'NASNetLarge', 'VGG19', 'InceptionV3', 'Xception'
]

# Iterate over each model
for model_name in model_names:
    print(f"Training model: {model_name}")

    # Get the model and preprocessing function
    model, preprocess_input = get_pretrained_model(model_name, (*image_size, 3), num_classes)

    # Create data generators with the specific preprocessing function
    train_generator = create_datagen(preprocess_input)
    validation_generator = create_datagen(preprocess_input)

    train_dataset = create_dataset(train_generator, train_df, train_dir, image_size)
    validation_dataset = create_dataset(validation_generator, validation_df, validation_dir, image_size)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        train_dataset,
        epochs=10,
        validation_data=validation_dataset
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(validation_dataset)
    print(f"Test accuracy for {model_name}: {accuracy*100:.2f}%")

   # Calculate additional metrics
y_true = validation_dataset.classes
y_pred = model.predict(validation_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted class labels

precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

# Use y_pred_classes (1D array) for AUC calculation
auc = roc_auc_score(y_true, y_pred_classes, multi_class='ovr')

print(f"Metrics for {model_name}:")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}, AUC: {auc:.2f}\n")
