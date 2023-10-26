import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
tf.get_logger().setLevel('ERROR')

# def extend_frames(frames, target_length):
#     if not frames:
#         return []
#
#     extended_frames = []
#     current_length = len(frames)
#
#     while len(extended_frames) < target_length:
#         # 현재 frames의 마지막부터 처음까지 역순으로 반복하는 index를 계산합니다.
#         # 예를 들어, frames가 [1, 2, 3, 4, 5] 라면 [4, 3, 2, 1] 순서로 index를 추가합니다.
#         for idx in range(current_length - 2, -1, -1):
#             extended_frames.append(frames[idx])
#             if len(extended_frames) == target_length:
#                 break
#
#     return extended_frames



# Define directories
base_dir = 'D:/Jiwoon/dataset/SCVD/videos/'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Function to load video frames using OpenCV
def load_video_frames(video_path, num_frames, img_height, img_width):
    if not isinstance(video_path, str):
        video_path = video_path.decode("utf-8")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_height, img_width))
        frames.append(frame)
    cap.release()
    frames = np.stack(frames)
    return frames

# Generator function for the dataset

def extend_frames(frames, target_length):
    if len(frames) == 0:
        return []

    # 현재 frames의 길이를 확인합니다.
    current_length = len(frames)

    # 만약 현재 프레임의 길이가 이미 target_length 이상이라면, 초과하는 프레임을 잘라냅니다.
    if current_length >= target_length:
        return frames[:target_length]

    # 확장된 프레임을 저장하기 위한 리스트를 준비합니다.
    # extended_frames = frames.copy()
    extended_frames = np.copy(frames)
    # extended_frames의 길이가 target_length에 도달할 때까지 프레임을 확장합니다.
    while len(extended_frames) < target_length:
        # 현재 frames의 마지막부터 처음까지 역순으로 프레임을 추가합니다.
        for idx in range(current_length - 2, -1, -1):
            # extended_frames.append(frames[idx])
            extended_frames = np.append(extended_frames, [frames[idx]], axis=0)

            if len(extended_frames) == target_length:
                break

    return extended_frames
def video_generator(directory):
    # labels = ['fight', 'nonfight']
    # for label in labels:
    #     label_dir = os.path.join(directory, label)
    #     for video_file in os.listdir(label_dir):
    #         video_path = os.path.join(label_dir, video_file)
    #         video_frames = load_video_frames(video_path)
    #         preprocessed_video = [tf.image.resize(frame, [256, 256]) for frame in video_frames]
    #         preprocessed_video = [frame / 255.0 for frame in preprocessed_video]  # Normalize
    #         yield tf.stack(preprocessed_video), labels.index(label)
    labels = ['fight', 'nonfight']
    for label in labels:
        label_dir = os.path.join(directory, label)
        for video_file in os.listdir(label_dir):
            video_path = os.path.join(label_dir, video_file)
            video_frames = load_video_frames(video_path, 300, 256, 256)
            video_frames = extend_frames(video_frames, 300)
            preprocessed_video = [tf.image.resize(frame, [256, 256]) for frame in video_frames]
            preprocessed_video = [frame / 255.0 for frame in preprocessed_video]  # Normalize
            yield tf.stack(preprocessed_video), labels.index(label)

# Create a tf.data.Dataset using the generator
def create_dataset(directory):
    return tf.data.Dataset.from_generator(
        lambda: video_generator(directory),
        output_signature=(
            tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

num_classes = 100
input_shape = (256, 256, 3)

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
train_dataset = create_dataset(train_dir)  # You can adjust batch size if needed
val_dataset = create_dataset(val_dir)

print(train_dataset)
for x_batch, y_batch in train_dataset.take(1):
    print(x_batch.shape)  # 입력 데이터의 형태를 출력합니다.
    print(y_batch.shape)
# print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
# print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 256  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# data_augmentation.layers[0].adapt(x_train)
for x_batch, _ in train_dataset.take(1):
    data_augmentation.layers[0].adapt(x_batch)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        if len(images.shape) == 3:
            images = tf.expand_dims(images, 0)

        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

plt.figure(figsize=(4, 4))
# image = x_train[np.random.choice(range(x_train.shape[0]))]
for x_batch, _ in train_dataset.take(1):
    image = x_batch[0][0][tf.newaxis, ...]  # 첫 번째 배치의 첫 번째 프레임만 선택하고 차원을 추가합니다.
resized_image = tf.image.resize(image, size=(image_size, image_size))
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class DataAugmentationLayer(layers.Layer):
    def __init__(self, data_augmentation, **kwargs):
        super(DataAugmentationLayer, self).__init__(**kwargs)
        self.data_augmentation = data_augmentation

    def call(self, inputs):
        # 입력 텐서의 형태를 구합니다.
        inputs_shape = tf.shape(inputs)
        batch_size, num_frames = inputs_shape[0], inputs_shape[1]

        # 데이터 증강된 프레임을 저장하기 위한 TensorArray를 초기화합니다.
        augmented_frames_array = tf.TensorArray(dtype=tf.float32, size=num_frames)

        # 각 프레임에 대해 데이터 증강을 적용합니다.
        for i in tf.range(num_frames):
            augmented_frame = self.data_augmentation(inputs[:, i])
            augmented_frames_array = augmented_frames_array.write(i, augmented_frame)

        augmented = augmented_frames_array.stack()
        augmented = tf.transpose(augmented, [1, 0, 2, 3, 4])

        return augmented

def create_vit_classifier():
    # inputs = layers.Input(shape=input_shape)
    inputs = layers.Input(shape=(None, 256, 256, 3))
    # Augment data.
    # augmented = data_augmentation(inputs)
    # augmented = tf.map_fn(data_augmentation, inputs)
    augmented = DataAugmentationLayer(data_augmentation)(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(1, activation='sigmoid')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            # keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    checkpoint_filepath = "./tmp/checkpoint/"
    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    # history = model.fit(
    #     x=x_train,
    #     y=y_train,
    #     batch_size=batch_size,
    #     epochs=num_epochs,
    #     validation_split=0.1,
    #     callbacks=[checkpoint_callback],
    # )
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[checkpoint_callback]
    )

    model.load_weights(checkpoint_filepath)

    # _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    _, accuracy, auc = model.evaluate(val_dataset)
    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")
    print(f"Validation AUC: {round(auc * 100, 2)}%")

    return history


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)







#
# callbacks = [
#     ModelCheckpoint('best_weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1),
#     EarlyStopping(patience=10, monitor='val_loss', mode='min', verbose=1),
#     ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1, monitor='val_loss', mode='min'),
#     TensorBoard(log_dir='./logs')
# ]
#
#
# vit_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                   metrics=metrics)
#
# # Create train and val datasets
# train_dataset = create_dataset(train_dir).batch(32)  # You can adjust batch size if needed
# val_dataset = create_dataset(val_dir).batch(32)
#
# history = vit_model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=callbacks)
#
#
# # Accuracy & Loss 그래프 그리기
# plt.figure(figsize=(14, 5))
#
# # Accuracy 그래프
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# # Loss 그래프
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# # AUC 그래프
# plt.figure(figsize=(7, 5))
# plt.plot(history.history['auc'], label='Train AUC')
# plt.plot(history.history['val_auc'], label='Validation AUC')
# plt.title('Training and Validation AUC')
# plt.xlabel('Epoch')
# plt.ylabel('AUC')
# plt.legend()
# plt.show()