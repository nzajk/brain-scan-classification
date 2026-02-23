import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.utils import shuffle
from tensorflow.keras.applications.efficientnet import preprocess_input

image_size_x, image_size_y = 224, 224

encode = {'notumor': 0, 'meningioma': 1, 'glioma': 2, 'pituitary': 3}
decode = {0: 'notumor', 1: 'meningioma', 2: 'glioma', 3: 'pituitary'}

train_dataset = 'src/backend/model/dataset/training/'
test_dataset  = 'src/backend/model/dataset/testing/'

# applied only to training images to artificially expand the dataset and expose
# the model to varied orientations/brightnesses, reducing overfitting.
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="augmentation")


def preprocess_image(img_path):
    if isinstance(img_path, np.ndarray):
        img = img_path
    else:
        img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size_x, image_size_y))
    img = img.astype(np.float32)
    img = preprocess_input(img)

    return img


def load_data(dataset_path):
    x_data = []
    y_data = []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            img = preprocess_image(file_path)
            x_data.append(img)
            y_data.append(encode[folder])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data, y_data = shuffle(x_data, y_data, random_state=101)
    return x_data, y_data


x_train, y_train = load_data(train_dataset)
x_test,  y_test  = load_data(test_dataset)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test  = tf.keras.utils.to_categorical(y_test,  num_classes=4)

# if one tumor class has far fewer samples it can get ignored; weighting
# compensates so every class contributes equally to the loss.
raw_labels  = np.argmax(y_train, axis=1)
class_counts = np.bincount(raw_labels)
total        = len(raw_labels)
class_weights = {i: total / (len(class_counts) * count)
                 for i, count in enumerate(class_counts)}

# EfficientNetB0 backbone is kept frozen initially so only the new head trains.
# this avoids corrupting ImageNet weights before the head has stabilised.
efficient_net = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(image_size_x, image_size_y, 3)
)
efficient_net.trainable = False          # frozen backbone

inputs = tf.keras.Input(shape=(image_size_x, image_size_y, 3))
x = augmentation(inputs)                # augmentation inside the model
x = efficient_net(x, training=False)    # backbone
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    # stop training when val_loss stops improving (patience = 5 epochs)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    ),
    # reduce LR when training plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
    # save only the best checkpoint
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
]

# train the head only
history_phase1 = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(x_test, y_test),
    class_weight=class_weights,
    callbacks=callbacks,
)

# fine-tune the top layers of the backbone
# unfreeze the last 60 layers of EfficientNetB0 and train at a much lower LR
# so we refine features without destroying pre-trained representations.
efficient_net.trainable = True
for layer in efficient_net.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),   # much lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_phase2 = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(x_test, y_test),
    class_weight=class_weights,
    callbacks=callbacks,
)

model.save('model.keras')
print("\nTraining complete. Model saved to model.keras")