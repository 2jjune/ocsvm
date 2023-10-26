import os
import cv2
import torch
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn import svm
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import svm
import torch.nn as nn
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from models.i3d.extract_i3d_custom import ExtractI3D
from utils.utils import build_cfg_path
from omegaconf import OmegaConf
from transformers import ViTForImageClassification, ViTConfig
import cnn_lstm_cbam
import torch.nn.functional as F
import pandas as pd
from transformers import SwinConfig, SwinModel
from GCVit.models import gc_vit
import matplotlib.pyplot as plt
from PIL import Image
from timeception.nets import timeception_pytorch
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_dir = 'D:/Jiwoon/dataset/two_stream_ubi_/data/'
data_dir = 'D:/Jiwoon/dataset/SCVD/videos/'
# data_dir = 'D:/Jiwoon/dataset/UBI_FIGHTS/vit/'
# data_dir = 'D:/Jiwoon/dataset/RWF_2000_Dataset/'
batch_size = 1
lr = 1e-4
num_epochs = 500
Image_size = 256

class VideoRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.transform = transforms.RandomHorizontalFlip(p)

    def __call__(self, frames):
        flip = np.random.rand() < self.p
        if flip:
            frames = torch.stack([self.transform(frame) for frame in frames])
        return frames


class VideoColorJitter(object):
    def __init__(self, brightness=0.3, contrast=0.2, saturation=0, hue=0):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, frames):
        transform = self.transform.get_params(self.transform.brightness,
                                              self.transform.contrast,
                                              self.transform.saturation,
                                              self.transform.hue)
        frames = torch.stack([transform(frame) for frame in frames])
        return frames


# 그리고 이제 이 두 변환을 모두 포함하는 변환을 만들 수 있습니다:
# transform = transforms.Compose([
#     VideoRandomHorizontalFlip(),
#     VideoColorJitter(),
#     transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames])),
# ])




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
transform = transforms.Compose([
        # VideoRandomHorizontalFlip(),
        # VideoColorJitter(),
        transforms.ToPILImage(),
        transforms.Resize((Image_size, Image_size)),#224
        transforms.ToTensor(),
        # transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames])),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=transform, annotations_dir="D:/Jiwoon/dataset/UBI_FIGHTS/annotation"):
        self.data_dir = data_dir
        self.transform = transform
        self.annotations_dir = annotations_dir

        # 동영상 파일 목록
        self.files = []
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            for filename in os.listdir(label_dir):
                self.files.append((os.path.join(label_dir, filename), label))

        # 클래스 인덱스
        self.class_indices = {'fight': 0, 'nonfight': 1}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 동영상 파일
        # print("self.files: ", self.files)
        filepath, label = self.files[idx]
        std_frame = 300
        max_frame = 150
        # 동영상 프레임 로드
        # print('filepath : ', filepath)
        print(filepath)
        cap = cv2.VideoCapture(filepath)
        frames = []
        frame_count = 0

        if 'val/fight' in filepath:
            print('fight!!')
            # filename = os.path.basename(filepath).replace('.mp4', '.csv')
            # csv_path = os.path.join(self.annotations_dir, filename)
            # annotations = pd.read_csv(csv_path, header=None)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # print(frame_count)
                # if annotations.iloc[frame_count, 0] == 1:
                if frame_count % 1 == 0:  # 20
                    # seg_frame = preprocess(frame).to(device)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = transform(frame)

                    # input_seg_frame = seg_frame.unsqueeze(0)
                    # seg_frame = segmentation_model(input_seg_frame)['out'][0].argmax(0).byte()
                    # seg_frame = transform(seg_frame).squeeze(0)
                    # print(frame.shape, frame.max(), seg_frame.max()*255)

                    # for channel in range(3):
                    #     frame[channel] += seg_frame*3

                    # 값이 255를 초과하는 경우 255로 클리핑
                    # frame = torch.clamp(frame, 0, 1.)
                    frames.append(frame)
                    if len(frames) >= max_frame:
                        break
                frame_count+=1

        elif 'nonfight' in filepath:
            print('nonfight~~')
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % 1 == 0:#20
                    # seg_frame = preprocess(frame).to(device)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = transform(frame)

                    # input_seg_frame=seg_frame.unsqueeze(0)
                    # seg_frame = segmentation_model(input_seg_frame)['out'][0].argmax(0).byte()
                    # seg_frame = transform(seg_frame).squeeze(0)
                    # print(frame.shape, frame.max(), frame.mean(), seg_frame.max())

                    # for channel in range(3):
                        # print(frame[channel].max(), seg_frame.max())
                        # frame[channel] += seg_frame*3

                    # 값이 255를 초과하는 경우 255로 클리핑
                    # frame = torch.clamp(frame, 0, 1.)

                    frames.append(frame)
                    if len(frames)>=max_frame:#14400
                        break
                frame_count+=1
        if len(frames)<max_frame:
            frames = extend_frames(frames,max_frame)
        cap.release()
        # frames = np.array(frames)
        frames = torch.stack(frames)

        print(len(frames))

        label_idx = self.class_indices[label]

        return frames, label_idx

def collate_fn(batch):
    frames, labels = zip(*batch)
    frames = [frame for frame in frames]
    frames = torch.stack(frames)#([1, 2000, 360, 640, 3])
    # frames = frames.permute(0,1,4,2,3)
    labels = torch.tensor(labels)
    return frames, labels

def collate_fn22(batch):
    frames, labels = zip(*batch)
    # frames = [torch.stack(frame) for frame in frames]
    # frames = [torch.stack(p) for patch in frames for p in patch]

    frames_stacked = []
    for i in range(4):  # 4개 패치를 대상으로 반복
        patch = [video_patches[i] for video_patches in frames]  # 각 비디오에서 i번째 패치를 모아 새로운 리스트 생성
        frames_stacked.append(torch.stack(patch))
    frames = torch.stack(frames_stacked)
    frames = frames.permute(1, 0, 2, 3, 4, 5)
    # frames = [torch.stack(video_patches) for video_patches in frames]  # 비디오 별로 패치 스택
    # frames = torch.stack(frames)
    labels = torch.tensor(labels)
    return frames, labels


train_dataset = VideoDataset(data_dir + 'train/', transform=transform)
val_dataset = VideoDataset(data_dir + 'val/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)


feature_type = 'i3d'
args = OmegaConf.load(build_cfg_path(feature_type))
args.flow_type = 'raft'

extractor = ExtractI3D(args)

import timm

def extract_vit(X):
    X_tensor = X.to(device)
    total_feature = []
    lstm_feature = []
    with torch.no_grad():
        # X_tensor = X_tensor.permute(0,2,1,3,4)
        # X_tensor_frame = X_tensor.float()
        # lstm_feature_array = cnn_lstm_model(X_tensor)
        # lstm_feature_array = lstm_feature_array.view(lstm_feature_array.size(0), -1).cpu().detach().numpy()
        # print('lstm', lstm_feature_array.shape)

        #---------------------2d
        for X_tensor_batch in X_tensor:
            # print('X tensor', X_tensor.shape)
            # X_flattened = X_tensor_batch.view(X_tensor_batch.shape[0], -1)
            # X_tensor_batch = X_tensor_batch.permute(1,0,2,3)
            print(X_tensor_batch.shape)
            features_array = swin_model(X_tensor_batch).last_hidden_state
            # features_array = swin_model(X_tensor_batch)
            print('bebevit', features_array.shape)
            # features_array = features_array.transpose(1,2)
            # features_array = F.avg_pool1d(features_array, kernel_size=9, stride=64)
            # features_array = features_array.transpose(1, 2)
            # features_array = timeception_model(features_array)
            features_array = torch.mean(features_array, dim=[1])
            # features_array = features_array.unsqueeze(0)
            # print('vit', features_array.shape)
            linear_layer = nn.Linear(1024, 768).to(device)
            features_transformed = linear_layer(features_array)
            # features_part1 = features_array[:, :512]
            # features_part2 = features_array[:, 512:]
            # features_array1 = transform_model(features_part1)
            # features_array2 = transform_model(features_part2)
            # features_array= (features_array1[0] + features_array2[0]) / 2.0

            attention_mask = torch.ones(1, features_transformed.shape[0], dtype=torch.long)  # Modified this line
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=next(transform_model.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            extended_attention_mask = extended_attention_mask.to(device)
            # BertModel의 encoder 속성 사용
            features_array = transform_model.encoder(features_transformed.unsqueeze(0), attention_mask=extended_attention_mask)
            features_array = features_array.last_hidden_state.squeeze(0)
            # print('vit', features_array.keys())
            print('vit', features_array.shape)
            # features_array = features_array.view(-1).cpu().detach()
            features_array = features_array.view(features_array.size(0), -1).cpu().detach().numpy()
            # features_array = features_array.view(features_array.size(0), -1)
            # features_array = features_array.reshape(-1, 2048*std_frame)
            # features_array = torch.tensor(features_array)
            total_feature.extend(features_array)
        # lstm_feature.extend(lstm_feature_array)

        #-----------------------for video
        # print(X_tensor.shape)
        # X_tensor = X_tensor.permute(0,2,1,3,4)
        #
        # features_array = videoswin_model(X_tensor)
        # print('bebevit', features_array.shape)
        #
        # # features_array = features_array.transpose(1,2)
        # # features_array = F.avg_pool1d(features_array, kernel_size=2, stride=4)
        # # features_array = features_array.transpose(1, 2)
        # features_array = torch.mean(features_array, dim=[2, 3])
        # print('vit', features_array.shape)
        # # features_array = features_array.view(-1).cpu().detach()
        # features_array = features_array.view(features_array.size(0), -1).cpu().detach().numpy()
        # # features_array = features_array.view(features_array.size(0), -1)
        # # features_array = features_array.reshape(-1, 2048*std_frame)
        # # features_array = torch.tensor(features_array)
        # total_feature.extend(features_array)

    return total_feature, lstm_feature


def extract_resnet_2d(X):
    # 이미지를 PyTorch Tensor로 변환
    # X_tensor = torch.stack(X).to(device)
    X_tensor = X.to(device)
    # print(X_tensor.shape)#torch.Size([1, 100, 3, 224, 224])  torch.Size([1, 4, 2000, 3, 224, 224])
    std_frame = 2000
    num_slices = X_tensor.shape[2] // std_frame
    batch = X_tensor.shape[0]
    total_feature = []

    resnet_model = models.video.r3d_18(pretrained=True).to(device)
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])  # 마지막 fully connected layer를 제외한 모델 가져오기
    resnet_model.eval()

    with torch.no_grad():
        X_tensor = X_tensor.permute(0,2,1,3,4)
        X_tensor_frame = X_tensor.float()
        features_array = resnet_model(X_tensor_frame)
        # features_array = features_array.view(-1).cpu().detach()
        features_array = features_array.view(features_array.size(0), -1).cpu().detach().numpy()
        # features_array = features_array.reshape(-1, 2048*std_frame)
        # features_array = torch.tensor(features_array)
        total_feature.extend(features_array)


    # for b in range(batch):
    #     X_tensor_frame = X_tensor[b]
    #     # print(123123123,X_tensor_frame.shape)
    #     # print('xtensor', X_tensor_frame.shape)#([4, 100, 3, 224, 224])
    #
    #     # tensor = torch.from_numpy(X_tensor_frame.cpu().detach().numpy().astype(np.float32)).to(device)
    #     # 표준편차 이미지 생성
    #     # for i in range(num_slices):
    #     #     start_idx = i * std_frame
    #     #     end_idx = start_idx + std_frame
    #     #     for j in range(X_tensor_frame.shape[0]):
    #     #     # print('tensor', X_tensor_frame.size())  # torch.Size([100, 360, 640, 3])
    #     #         std_r = torch.std(X_tensor_frame[j, start_idx:end_idx, 0, :, :], dim=0, keepdim=True)
    #     #         std_g = torch.std(X_tensor_frame[j, start_idx:end_idx, 1, :, :], dim=0, keepdim=True)
    #     #         std_b = torch.std(X_tensor_frame[j, start_idx:end_idx, 2, :, :], dim=0, keepdim=True)
    #     #         # RGB 표준편차를 하나의 Tensor로 결합
    #     #         rgb = torch.cat([std_r, std_g, std_b])
    #     #         images[j].append(rgb)
    #
    #     # ResNet 모델 불러오기
    #     resnet_model = models.video.r3d_18(pretrained=True).to(device)
    #     # resnet_model = models.wide_resnet50_2(pretrained=True).to(device)
    #     # resnet_model = models.resnet50(pretrained=True).to(device)
    #     resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])  # 마지막 fully connected layer를 제외한 모델 가져오기
    #     # print(resnet_model)
    #     # 추출된 특성 벡터 계산
    #     resnet_model.eval()
    #
    #     # num_frames_per_segment = 20
    #     # num_segments = X_tensor_frame.shape[1] // num_frames_per_segment
    #     # images = [[], [], [], []]
    #     # for i in range(num_segments):
    #     #     start_idx = i * std_frame
    #     #     end_idx = start_idx + std_frame
    #     #
    #     #     for j in range(X_tensor_frame.shape[0]):
    #     #         # 각 채널별로 표준편차를 계산
    #     #         std_r = torch.std(X_tensor_frame[j, start_idx:end_idx, 0, :, :], dim=0, keepdim=True)
    #     #         std_g = torch.std(X_tensor_frame[j, start_idx:end_idx, 1, :, :], dim=0, keepdim=True)
    #     #         std_b = torch.std(X_tensor_frame[j, start_idx:end_idx, 2, :, :], dim=0, keepdim=True)
    #     #
    #     #         # RGB 표준편차를 하나의 Tensor로 결합
    #     #         rgb = torch.cat([std_r, std_g, std_b], dim=0)
    #     #         images[j].append(rgb)
    #     #
    #     # # 각 배치의 이미지를 하나의 Tensor로 결합
    #     # images = [torch.stack(batch_images) for batch_images in images]
    #
    #     # 리스트를 하나의 Tensor로 변환
    #     # X_tensor_frame = torch.stack(images)
    #
    #     with torch.no_grad():
    #         X_tensor_frame = X_tensor_frame.float()
    #         features_array = resnet_model(X_tensor_frame)
    #         # features_array = features_array.view(-1).cpu().detach()
    #         features_array = features_array.view(features_array.size(0), -1).cpu().detach().numpy()
    #         # features_array = features_array.reshape(-1, 2048*std_frame)
    #         # features_array = torch.tensor(features_array)
    #         total_feature.extend(features_array)
    #
    #
    #         # for f in range(X_tensor_frame.shape[1]):
    #         #     X_tensor_frame_split = X_tensor_frame[:, f, :, :, :]
    #         #     # print('bebebebesplitps;otibef', X_tensor_frame_split.size())
    #         #     X_tensor_frame_split = X_tensor_frame_split.squeeze(dim=1)
    #         #     # print('afafafafsplitps;oti', X_tensor_frame_split.size())
    #         #     '''
    #         #     # # images_tensor = torch.stack(images)
    #         #     # images_tensor = [torch.stack(image_list) for image_list in images]
    #         #     # images_tensor = torch.stack(images_tensor)
    #         #     # images_tensor = images_tensor.permute(0,2,1,3,4)
    #         #     # # features_array = resnet_model(images_tensor)
    #         #     # images_tensor = images_tensor.squeeze(dim=2)
    #         #     # # print('image tensor', images_tensor.size())#([4, 3, 2, 224, 224])
    #         #     '''
    #         #     features_array = resnet_model(X_tensor_frame_split)
    #         #     # features_array = features_array.view(-1).cpu().detach()
    #         #     features_array = features_array.view(features_array.size(0), -1).cpu().detach().numpy()
    #         #     # features_array = features_array.reshape(-1, 2048*std_frame)
    #         #     # features_array = torch.tensor(features_array)
    #         #     total_feature.extend(features_array)
    #
    #     # batch_size, num_frames, _, _, _ = features_array.shape
    #     # features_array = features_array.view(batch_size, -1).to(device)  # [B, F, C*H*W] 형태로 변환
    #     # features_array = features_array.view(batch_size, num_frames, -1).to(device)  # [B, F, C*H*W] 형태로 변환
    #     # features_array = features_array.view(batch_size, num_frames)  # [B, F, C*H*W] 형태로 변환
    return total_feature


def extract_resnet(X):
    # 이미지를 PyTorch Tensor로 변환
    # print(X.size())
    # X_tensor = torch.stack([transform(image) for image in X]).to(device)
    X_tensor = torch.stack(X).to(device)
    # print(X_tensor.size())
    X_tensor = X_tensor.permute(0,2,1,3,4)
    # print(X_tensor.size())

    # ResNet 모델 불러오기
    # resnet_model = models.resnet50(pretrained=True).to(device)
    resnet_model = models.video.r3d_18(pretrained=True).to(device)
    # resnet_model = models.video.s3d(pretrained=True).to(device)
    # resnet_model = models.video.mvit_v1_b(pretrained=True).to(device)
    # resnet_model = resnet_model.unsqueeze(0).expand_as(x)
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])  # 마지막 fully connected layer를 제외한 모델 가져오기
    # print(resnet_model)
    # 추출된 특성 벡터 계산
    resnet_model.eval()
    with torch.no_grad():
        features_array = resnet_model(X_tensor)
    batch_size, num_frames, _, _, _ = features_array.shape
    features_array = features_array.view(batch_size, -1).to(device)  # [B, F, C*H*W] 형태로 변환
    # features_array = features_array.view(batch_size, num_frames, -1).to(device)  # [B, F, C*H*W] 형태로 변환
    # features_array = features_array.view(batch_size, num_frames)  # [B, F, C*H*W] 형태로 변환
    return features_array.cpu().detach().numpy()

def extract_convlstm(X):
    X_tensor = torch.stack(X).to(device)
    # print(X_tensor.size())#torch.Size([1, 20, 3, 224, 224])

    convlstm = model.ConvLSTM(input_dim=3,
                     hidden_dim=[64, 64],
                     kernel_size=(3, 3),
                     num_layers=2,
                     batch_first=True,
                     bias=True,
                     return_all_layers=False).to(device)
    # convlstm = model.ConvLSTM(input_dim=3,
    #                  hidden_dim=16,
    #                  kernel_size=(3, 3),
    #                  num_layers=1,
    #                  batch_first=True,
    #                  bias=True,
    #                  return_all_layers=False).to(device)
    # Example:
    #     >> x = torch.rand((32, 10, 64, 128, 128))
    #     >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
    #     >> _, last_states = convlstm(x)
    #     >> h = last_states[0][0]  # 0 for layer index, 0 for h index

    convlstm.eval()
    with torch.no_grad():
        features_array = convlstm(X_tensor)

    # batch_size, num_frames, _, _, _ = features_array.shape
    # features_array = features_array.view(batch_size, -1).to(device)  # [B, F, C*H*W] 형태로 변환
    layer_output, last_state = features_array


    # return features_array.cpu().detach().numpy()
    return last_state


class TimeFeatureRelationNetwork(nn.Module):
    def __init__(self, input_channels=1024):
        super(TimeFeatureRelationNetwork, self).__init__()

        # Multi-scale time convolution layers
        self.conv1 = nn.Conv1d(input_channels, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_channels, 512, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_channels, 512, kernel_size=7, padding=3)

        self.relu = nn.ReLU()

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Adjusting the input tensor dimensions for 1D convolutions
        # We'll consider the dimension of 64 as a batch dimension temporarily
        x = x.permute(1, 2, 0).contiguous()

        # Apply multi-scale convolutions
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(x))
        out3 = self.relu(self.conv3(x))

        # Concatenate outputs from multi-scale convolutions
        out = torch.cat([out1, out2, out3], dim=1)

        # Apply global average pooling
        out = self.global_avg_pool(out).squeeze(2)

        # Adjusting the output tensor dimensions back to original format
        out = out.permute(1, 0).contiguous()

        return out


class BiLSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=8):
        super(BiLSTMFeatureExtractor, self).__init__()

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Adjusting the input tensor dimensions for LSTM
        # We'll consider the dimension of 64 as a batch dimension
        x = x.permute(1, 0, 2)

        # LSTM forward pass
        out, _ = self.bilstm(x)

        # Adjusting the output tensor dimensions back to original format
        out = out.permute(1, 0, 2)

        return out

# 변환 및 모델 설정



# # oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search
# oc_svm_clf = svm.OneClassSVM(gamma='scale', kernel='rbf', nu=0.08)  # Obtained using grid search
# if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search
# lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
oc_svm_clf = svm.OneClassSVM(gamma='auto', kernel='rbf', nu=0.55)
if_clf = IsolationForest(contamination=0.2, max_features=1.0, max_samples='auto', n_estimators=200)
lof = LocalOutlierFactor(n_neighbors=5, contamination=0.5, novelty=True)


segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
segmentation_model.eval()
# print([model for model in timm.list_models(pretrained=True) if model.startswith('vit')])
# vit_base_patch8_224  vit_huge_patch14_224
# vit_model = timm.create_model('vit_huge_patch14_224', pretrained=True).to(device)#vit_base_patch16_224 vit_large_patch16_224  vit_large_patch16_384  vit_huge_patch14_224_ijepa.in22k  vit_gigantic_patch16_224_ijepa.in22k
# vit_model.head = nn.Identity()
# vit_model = nn.Sequential(*list(vit_model.children())[:-1])
# vit_model.eval()

# from transformers import ConvNextConfig, ConvNextModel
# configuration = ConvNextConfig()
# convnext_model = ConvNextModel(configuration).from_pretrained("facebook/convnext-large-224").to(device)
# configuration = convnext_model.config
#
# convnext_model2 = models.convnext_base(pretrained=True).to(device)
# convnext_model2 = nn.Sequential(*list(convnext_model2.children())[:-1])


from transformers import SwinConfig, SwinModel
from transformers import Swinv2Config, Swinv2Model
'''256,384 22k base -- swinv2_base_patch4_window12_192_22k swinv2_base_patch4_window12_192_22k swinv2_large_patch4_window12_192_22k
1k base -- swinv2_base_patch4_window12to16_192to256_22kto1k_ft swinv2_base_patch4_window12to24_192to384_22kto1k_ft swinv2_large_patch4_window12to16_192to256_22kto1k_ft swinv2_large_patch4_window12to24_192to384_22kto1k_ft
'''
# Initializing a Swin microsoft/swin-tiny-patch4-window7-224 style configuration
configuration = Swinv2Config()
swin_model = Swinv2Model(configuration).from_pretrained("microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft").to(device)

# swin_model = gc_vit.gc_vit_large_384_21k(pretrained=False).to(device)
# swin_model = timm.models.create_model(
#         'gc_vit_large_512_21k',
#         pretrained=False,
#         # num_classes=args.num_classes,
#         in_chans=3,
#         global_pool=None,
#         scriptable=False,
#         # **args.model_kwargs,
#     )
# # torch.load(pretrained)['state_dict']
# # swin_model.load_state_dict(torch.load('gcvit_21k_large_512.pth.tar'))
# swin_model.load_state_dict(torch.load('gcvit_21k_large_384.pth.tar')['state_dict'])

# swin_model = timm.create_model('swinv2_base_patch4_window12to16_192to256_22kto1k_ft', pretrained=True).to(device)
# swin_model = nn.Sequential(*list(swin_model.children())[:-1])
swin_model.eval()
# BERT의 설정과 인코더 로드
from transformers import BertConfig
config = BertConfig()
model_name = "bert-base-uncased"
transform_model = BertModel(config).from_pretrained(model_name).to(device)
transform_model.eval()



# print(swin_model)
# configuration = swin_model.config
# swin_model = nn.Sequential(*list(swin_model.children())[:-2])
# swin_model2 = torch.load('mask_rcnn_swin_tiny_patch4_window7.pth')
# print(swin_model2)
# print(dir(models.video.swin_transformer))
# videoswin_model = models.video.swin_transformer.swin3d_b(pretrained=True).to(device)
# videoswin_model = models.video.swin_transformer.swin3d_b(weights="DEFAULT").to(device)
# videoswin_model = models.video.s3d(weights="DEFAULT").to(device)
# videoswin_model = nn.Sequential(*list(videoswin_model.children())[:-1])
# videoswin_model.eval()
# print(videoswin_model)
# import gcvit
# #['GCViT', 'GCViTBase', 'GCViTLarge', 'GCViTSmall', 'GCViTTiny', 'GCViTXTiny', 'GCViTXXTiny', '__builtins__', '__cached__', '__doc__', '__file__',
# # '__loader__', '__name__', '__package__', '__path__', '__spec__', '__version__', 'conv_utils', 'layers', 'models', 'utils', 'version']
# print(dir(gcvit))
# swin_model = gcvit.GCViTLarge(pretrain=True)#tensorflow
# print(swin_model)
# resnet_model = models.resnet50(pretrained=True).to(device)
# resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
# resnet_model.eval()
#
cnn_lstm_model = cnn_lstm_cbam.CNNLSTM(2).to(device)
# cnn_lstm_model = nn.Sequential(*list(cnn_lstm_model.children())[:-2])
cnn_lstm_model.eval()

# print('/'*100)
# input = torch.tensor(np.zeros((300,64,1024)), dtype=torch.float32)
# print(input.size())
timeception_model = BiLSTMFeatureExtractor().to(device)
# print(timeception_model)
# tesnor = timeception_model(input)
# print(tesnor.size())


# print(cnn_lstm_model)
vit_feature = []
lstm_feature = []
count = 0
for inputs, label in train_loader:#1 = nonfight    ([1, 250, 3, 224, 224])
    # features = extract_resnet_2d(inputs)
    # for inp in inputs:

    features, lstm_features = extract_vit(inputs)
    # print(features['pixel_values'].shape)#([250, 3, 224, 224])
    vit_feature.extend(features)
    # lstm_feature.extend(lstm_features)
    count += 1
    print('실행 {}, 남은영상 {}'.format(count, len(train_loader)-count))
# train_features = extractor.extract(train_loader)

# print(len(train_loader))#2
# print(np.array(res_feature).shape)#(200, 2048)
# print((train_features['rgb'] + train_features['flow']).shape)#4,1024


vit_feature = np.array(vit_feature).reshape(len(train_loader),-1)
# lstm_feature = np.array(lstm_feature).reshape(len(train_loader),-1)
# i3d_feature = (train_features['rgb'] + train_features['flow']).reshape(len(train_loader),-1)
# all_features = np.concatenate((vit_feature, i3d_feature),axis=1)


oc_svm_clf.fit(vit_feature)
if_clf.fit(vit_feature)
lof.fit(vit_feature)



oc_svm_preds = []
if_preds = []
lof_preds = []

all_features = []
val_labels = []
vit_feature = []
lstm_feature = []
count = 0
# val_features = extractor.extract(val_loader)
for inputs, label in val_loader:
#     # inputs = inputs.to(device)
#
#     # ResNet 특징 추출
#     # features = extract_resnet(inputs)
#     # features = extract_convlstm(inputs)
#     features = extract_resnet_2d(inputs)
#     all_features.extend(features)
#     features = extract_resnet_2d(inputs)
    features, lstm_features = extract_vit(inputs)
    # lstm_feature.extend(lstm_features)

    vit_feature.extend(features)
    val_labels.extend(label.numpy())
    count += 1
    print('실행 {}, 남은영상 {}'.format(count, len(val_loader) - count))
# # all_features = np.concatenate(all_features, axis=0)
# # all_features = imputer.fit_transform(all_features)

# all_features = val_features['flow']
# all_features = np.concatenate((val_features['rgb'], val_features['flow']), axis=1)
# all_features = val_features['rgb'] + val_features['flow']

# lstm_feature = np.array(lstm_feature).reshape(len(val_loader),-1)
vit_feature = np.array(vit_feature).reshape(len(val_loader),-1)
# i3d_feature = (val_features['rgb'] + val_features['flow']).reshape(len(val_loader),-1)
# all_features = np.concatenate((vit_feature, i3d_feature),axis=1)


# OneClassSVM 예측
oc_svm_pred = oc_svm_clf.predict(vit_feature)
oc_svm_preds.extend(oc_svm_pred)
oc_svm_preds = list(map(lambda x: 0 if x == -1 else x, oc_svm_preds))
oc_svm_preds_voted = np.array(oc_svm_preds)
# oc_svm_preds_voted = np.where(oc_svm_preds_voted.reshape(-1, 5).min(axis=1) == 0, 0, 1)

# num_images = len(oc_svm_preds) // 4
# oc_svm_preds_reshaped = oc_svm_preds.reshape(num_images, 4)
#
# oc_svm_preds_voted = np.apply_along_axis(
#     lambda x: np.argmax(np.bincount(x)), axis=1, arr=oc_svm_preds_reshaped)

# Isolation Forest 예측
if_pred = if_clf.predict(vit_feature)
if_preds.extend(if_pred)
if_preds = list(map(lambda x: 0 if x == -1 else x, if_preds))
if_preds_voted = np.array(if_preds)
# if_preds_voted = np.where(if_preds_voted.reshape(-1, 5).min(axis=1) == 0, 0, 1)
#
# num_images = len(if_preds) // 4
# if_preds_reshaped = if_preds.reshape(num_images, 4)
# if_preds_voted = np.apply_along_axis(
#     lambda x: np.argmax(np.bincount(x)), axis=1, arr=if_preds_reshaped)


lof_pred = lof.predict(vit_feature)
lof_preds.extend(lof_pred)
lof_preds = list(map(lambda x: 0 if x == -1 else x, lof_preds))
lof_preds_voted = np.array(lof_preds)
# lof_preds_voted = np.where(lof_preds_voted.reshape(-1, 5).min(axis=1) == 0, 0, 1)
# num_images = len(lof_preds) // 4
# lof_preds_reshaped = lof_preds.reshape(num_images, 4)
# lof_preds_voted = np.apply_along_axis(
#     lambda x: np.argmax(np.bincount(x)), axis=1, arr=lof_preds_reshaped)


correct_predictions = sum(1 for true, pred in zip(val_labels, oc_svm_preds_voted) if true == pred)
accuracy = correct_predictions / len(val_labels) * 100
print('shapahsahp', oc_svm_preds_voted.shape, np.array(val_labels).shape)
print("oc_svm   : ", oc_svm_preds_voted[20:80], "acc : ", accuracy)
roc_auc_train = roc_auc_score(val_labels, oc_svm_preds_voted)
print("oc_svm auc  : ", roc_auc_train)

correct_predictions = sum(1 for true, pred in zip(val_labels, if_preds_voted) if true == pred)
accuracy = correct_predictions / len(val_labels) * 100
print("if_preds : ", if_preds_voted[20:80], "acc : ", accuracy)
roc_auc_train = roc_auc_score(val_labels, if_preds_voted)
print("if_preds auc  : ", roc_auc_train)


correct_predictions = sum(1 for true, pred in zip(val_labels, lof_preds_voted) if true == pred)
accuracy = correct_predictions / len(val_labels) * 100
print("lof_pred : ", lof_preds_voted[20:80], "acc : ", accuracy)
roc_auc_train = roc_auc_score(val_labels, lof_preds_voted)
print("lof_pred auc  : ", roc_auc_train)


print('label    : ', np.array(val_labels)[20:80])