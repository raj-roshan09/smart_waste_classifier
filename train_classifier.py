# train_classifier.py
import os, json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# CONFIG
DATA_DIR = "datasets/classifier"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
OUT_MODEL = "models/waste_classifier.h5"
CLASS_MAP_JSON = "models/class_indices.json"

os.makedirs("models", exist_ok=True)

# DATA
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# Save class indices mapping (folder name -> index)
class_indices = train_gen.class_indices
with open(CLASS_MAP_JSON, "w") as f:
    json.dump(class_indices, f)
print("Saved class indices to", CLASS_MAP_JSON, class_indices)

NUM_CLASSES = train_gen.num_classes

# MODEL
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
out = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(OUT_MODEL, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
print("Training finished. Best model saved to", OUT_MODEL)

# Optional: Fine-tune some base layers
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
ft_history = model.fit(train_gen, validation_data=val_gen, epochs=6, callbacks=callbacks)
print("Fine-tune finished. Model saved to", OUT_MODEL)
