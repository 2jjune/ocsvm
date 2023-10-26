import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa
import glob, random, os, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from vit_keras import vit
import cv2


print('TensorFlow Version ' + tf.__version__)
def seed_everything(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'

seed_everything()

import warnings
warnings.filterwarnings("ignore")

image_size = 224
batch_size = 16
n_classes = 3
EPOCHS = 30
max_frame = 300
classes = {0 : "fight",
           1 : "nonfight"}

# def load_and_preprocess_video(video_path, label):
#     video = tf.io.read_file(video_path)
#     video_frames = tf.image.decode_video(video, max_frames=300)  # 최대 10000 프레임으로 제한
#
#     # 전처리
#     frames = tf.image.resize(video_frames, [image_size, image_size])  # 예시로 224x224로 리사이즈
#     frames = tf.cast(frames, tf.float32) / 255.0  # 픽셀 값을 [0,1] 범위로 정규화
#     return frames, tf.repeat(label, tf.shape(frames)[0])  # 각 프레임에 라벨을 할당
def extend_frames(frames, target_length):
    if not frames:
        return []

    extended_frames = []
    current_length = len(frames)

    while len(extended_frames) < target_length:
        # 현재 frames의 마지막부터 처음까지 역순으로 반복하는 index를 계산합니다.
        # 예를 들어, frames가 [1, 2, 3, 4, 5] 라면 [4, 3, 2, 1] 순서로 index를 추가합니다.
        for idx in range(current_length - 2, -1, -1):
            extended_frames.append(frames[idx])
            if len(extended_frames) == target_length:
                break

    return extended_frames
def load_and_preprocess_video_wrapper(video_path, label):
    frames, labels = tf.py_function(load_and_preprocess_video, [video_path, label], [tf.float32, tf.int64])

    # 튜플의 각 요소에 대한 형태와 데이터 타입을 설정
    frames.set_shape([None, image_size, image_size, 3])
    labels.set_shape([None])

    return tf.data.Dataset.from_tensor_slices((frames, labels))

def load_and_preprocess_video(video_path, label):
    if isinstance(video_path, tf.Tensor):
        video_path = video_path.numpy().decode("utf-8")
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size))
        frames.append(frame)
        if len(frames) >= max_frame:
            break
    cap.release()
    if len(frames) < max_frame:
        frames = extend_frames(frames, max_frame)

    frames = np.stack(frames) / 255.0
    return frames, tf.repeat(tf.cast(label, tf.int64), frames.shape[0])

base_dir = 'D:/Jiwoon/dataset/SCVD/videos/'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# 경로 및 레이블 로드
train_files = glob.glob(train_dir + "/*/*.mp4")
val_files = glob.glob(val_dir + "/*/*.mp4")

train_labels = [0 if "nonfight" in x else 1 for x in train_files]
val_labels = [0 if "nonfight" in x else 1 for x in val_files]

train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

# 동영상 데이터를 로드 및 전처리
# train_gen = train_ds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices(load_and_preprocess_video(x, y)))
# val_gen = val_ds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices(load_and_preprocess_video(x, y)))
train_gen = train_ds.flat_map(load_and_preprocess_video_wrapper)
val_gen = val_ds.flat_map(load_and_preprocess_video_wrapper)
train_gen = train_gen.shuffle(1024).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
val_gen = val_gen.batch(batch_size).prefetch(tf.data.AUTOTUNE)


vit_model = vit.vit_b16(
        image_size = image_size,
        activation = 'sigmoid',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = 1)

class Patches(L.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation = tfa.activations.gelu),
        tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
        tf.keras.layers.Dense(1, 'sigmoid')
    ],
    name = 'vision_transformer')

model.summary()

warnings.filterwarnings("ignore")

learning_rate = 1e-4
weight_decay = 0.0001
optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

model.compile(
        optimizer=optimizer,
        # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            # keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
vit_model.summary()
total_train_samples = sum([len(load_and_preprocess_video(f, l)[0]) for f, l in zip(train_files, train_labels)])
total_val_samples = sum([len(load_and_preprocess_video(f, l)[0]) for f, l in zip(val_files, val_labels)])

# 스텝 크기 계산
STEP_SIZE_TRAIN = total_train_samples // batch_size
STEP_SIZE_VALID = total_val_samples // batch_size

# STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
# STEP_SIZE_VALID = val_gen.n // val_gen.batch_size



early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True, verbose = 1)

model.fit(x = train_gen,
          steps_per_epoch = STEP_SIZE_TRAIN,
          validation_data = val_gen,
          validation_steps = STEP_SIZE_VALID,
          epochs = EPOCHS,
          callbacks = early_stopping_callbacks)

