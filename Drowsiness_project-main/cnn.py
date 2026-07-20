import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# 1️⃣ Settings
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = r"C:\Users\dandu\Downloads\capstone\dataset"

# ----------------------------
# 2️⃣ Data generators
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False   # IMPORTANT for correct evaluation
)

print("Class labels:", train_generator.class_indices)

# ----------------------------
# 3️⃣ Load pretrained MobileNetV2
# ----------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# 4️⃣ Early stopping
# ----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ----------------------------
# 5️⃣ Train
# ----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ----------------------------
# 6️⃣ EVALUATION METRICS
# ----------------------------
print("\n📊 Evaluating model...\n")

val_generator.reset()
pred = model.predict(val_generator)
pred_classes = np.argmax(pred, axis=1)
true_classes = val_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_classes, pred_classes)
print("Confusion Matrix:\n", cm)

# Precision, Recall, F1-score
class_labels = list(val_generator.class_indices.keys())
report = classification_report(true_classes, pred_classes, target_names=class_labels)
print("\nClassification Report:\n", report)

# ----------------------------
# 7️⃣ Save model
# ----------------------------
MODEL_PATH = os.path.join(os.getcwd(), "drowsiness_model.keras")
model.save(MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")

# ----------------------------
# 8️⃣ Convert to TFLite
# ----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

TFLITE_PATH = os.path.join(os.getcwd(), "drowsiness_model.tflite")
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {TFLITE_PATH}")