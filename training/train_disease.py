import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def main():
    base_dir = r"c:\Users\abhay\Desktop\smart_agriculture_project"
    data_dir = os.path.join(base_dir, "dataset", "plant_disease", "PlantVillage")
    model_dir = os.path.join(base_dir, "models")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Class mapping to match app.py strictly
    target_classes = [
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___healthy",
        "Potato___Late_blight",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_healthy",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
        "Tomato__Tomato_mosaic_virus",
        "Tomato__Tomato_YellowLeaf__Curl_Virus"
    ]
    
    img_size = (224, 224)
    batch_size = 32
    
    print("Loading image dataset...")
    # MobileNetV2 expects [-1, 1] input typically, but the old code divided by 255.0.
    # To strictly keep prediction compatible with existing app.py which does / 255.0:
    # We will rescale by 1./255.
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        class_names=target_classes,
        label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        class_names=target_classes,
        label_mode='categorical'
    )
    
    # Preprocessing
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Performance caching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print("Building MobileNetV2 model...")
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False # Freeze base model for faster training
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(15, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model_path = os.path.join(model_dir, "plant_disease_model.h5")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy')
    ]
    
    print("Starting perfectly trained transfer learning for 3 epochs (optimized for CPU)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        steps_per_epoch=50,
        validation_steps=10,
        callbacks=callbacks
    )
    
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    main()
