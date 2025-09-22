import os
import glob
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO  # 가정한 모듈명, 실제로는 해당하는 YOLO 모듈로 대체해야 함
import cv2
import torch
import copy
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import inspect
 

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet_v2.features
        for param in self.features.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        return self.features(x)

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='batchnorm', apply_dropout=False):
        super(UpSampleBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if norm_type.lower() == 'batchnorm':
            layers.append(nn.BatchNorm2d(out_channels))
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.upsample = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.upsample(x)
class MobileNetV2_UNet(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileNetV2_UNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = nn.Sequential(
            UpSampleBlock(1280, 512, apply_dropout=True),
            UpSampleBlock(512, 256),
            UpSampleBlock(256, 128),
            UpSampleBlock(128, 64),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # 출력 크기 조정
        x = F.interpolate(x, size=(960, 544), mode='bilinear', align_corners=False)
        return x
    
class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = transforms.Resize(self.size)(image)
        mask = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST)(mask)
        return image, mask
    
    

class CustomTransform:
    def __init__(self):
        self.resize = transforms.Resize((960, 544))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, mask):
        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)

        # 마스크는 크기만 조정하고, 텐서로 변환합니다. Normalize는 적용하지 않습니다.
        mask = self.resize(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask
    
# 데이터셋 정의
class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '_mask.png'))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
    
    
    
dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# 인자 파싱을 위한 설정
parser = argparse.ArgumentParser(description='모델 실행을 위한 인자 파싱')
parser.add_argument('--data_dir', required=True, help='이미지 폴더 경로를 입력하세요.')
parser.add_argument('--result_dir', required=True, help='분석 결과가 저장될 폴더 경로를 입력하세요.')
args = parser.parse_args()


# 모델 인스턴스 생성
model = MobileNetV2_UNet(num_classes=4)  # 클래스 수에 맞게 조정해야 할 수 있습니다.

# 모델을 CPU나 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model__path = dir + '/model/best_model'
# 모델 가중치 로드 (학습된 모델 가중치 경로 지정 필요)
model.load_state_dict(torch.load(os.path.join(args.data_dir, model__path), map_location=device))

# 모델을 평가 모드로 설정
model.eval()

# predicted_mask 디렉토리 설정 및 생성
predicted_mask_dir = os.path.join(args.result_dir, 'predicted_mask')
if not os.path.exists(predicted_mask_dir):
    os.makedirs(predicted_mask_dir)
    
    
def mask_to_color_image(mask):
    """예측된 마스크를 색상 이미지로 변환합니다."""
    colors = np.array([
        [0, 0, 0],       # 검정
        [255, 0, 0],     # 빨강
        [0, 0, 255],     # 파랑
        [255, 255, 0]    # 노랑
    ], dtype=np.uint8)

    color_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    for class_idx in range(4):
        color_mask[mask[0] == class_idx] = colors[class_idx] 
    return color_mask


def visualize_prediction_with_image(model, image_path, device, result_dir, predicted_mask_dir):
    # 이미지 전처리 및 모델을 통한 예측
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # 원본 이미지의 크기를 저장
    
    preprocess = transforms.Compose([
        transforms.Resize((960, 544)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        # 예측된 마스크의 크기를 원본 이미지의 크기로 조정
        output = F.interpolate(output, size=original_size[::-1], mode='bilinear', align_corners=False)
        predicted_mask = torch.argmax(output, dim=1).cpu()
    
    color_mask = mask_to_color_image(predicted_mask.numpy())
    
    # 원본 이미지의 크기로 색상 마스크 이미지 조정을 제거
    color_mask_resized = Image.fromarray(color_mask)  # .resize(original_size, Image.NEAREST)를 제거

    # 저장할 파일명과 경로 설정
    image_name = os.path.basename(image_path).split('.')[0] + '_predicted_mask.png'
    save_path = os.path.join(predicted_mask_dir, image_name)  # Here, use predicted_mask_dir instead of result_dir

    # 예측된 색상 마스크 시각화 및 저장
    color_mask_resized.save(save_path)
    print(f"Saved predicted mask to {save_path}")
    
    # 저장된 파일 경로 반환
    return save_path





# 이미지 폴더에서 이미지 파일 경로를 가져오기
file_paths = glob.glob(os.path.join(args.data_dir, '*.jpg'))

# 분석 결과 디렉토리가 없으면 생성하기
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

# 예측 및 라벨 데이터 저장
for image_path in file_paths:
    # 이미지 이름 추출 (확장자 제외)
    image_name = os.path.basename(image_path).split('.')[0]

    # 마스크 이미지 로드 및 처리
    mask_image_path = visualize_prediction_with_image(model, image_path, device, args.result_dir, predicted_mask_dir)
    mask_image = Image.open(mask_image_path)
    mask_array = np.array(mask_image)

    # 마스크 이미지가 RGBA라면, 알파 채널 제거
    if mask_array.shape[-1] == 4:
        mask_array = mask_array[..., :3]

    # YOLO 모델로 객체 탐지 실행
    model_yolo = YOLO(os.path.join(args.data_dir, (dir + '/model/best.pt')))
    results = model_yolo(image_path)

    # 탐지된 객체들의 라벨 정보 추출
    boxes = results[0].boxes
    cls = boxes.cls  # 객체의 클래스 ID
    confs = boxes.conf  # 객체의 신뢰도(confidence)
    xywh = boxes.xywh  # 객체의 바운딩 박스 (x_center, y_center, width, height)
    # 이미지의 너비와 높이 설정
    img_width, img_height = 1920, 1080

    # 라벨 정보를 텍스트 파일로 저장 (이미지 이름을 기반으로 함)
    label_text_path = os.path.join(args.result_dir, f"{image_name}.txt")
    with open(label_text_path, 'w') as file:
        for i in range(len(cls)):
            # 바운딩 박스 좌표를 이미지 크기로 나누어 정규화
            x = xywh[i][0].item() / img_width
            y = xywh[i][1].item() / img_height
            w = xywh[i][2].item() / img_width
            h = xywh[i][3].item() / img_height
            # 해당 좌표의 픽셀 값 확인
            x_center_int = int(round(xywh[i][0].item()))
            y_lower_half_int = int(round(xywh[i][1].item() + (xywh[i][3].item() / 2)))
            pixel_value = mask_array[y_lower_half_int, x_center_int]
            # 해당 좌표의 픽셀이 빨간색(도로)인지 확인
            is_in_red_area = np.all(pixel_value == [255, 0, 0])
            # 정규화된 좌표, 클래스 ID 및 추가 정보를 파일에 쓰기
            name = ''
            if cls[i].item() == 0:
                name = 'person'
            elif cls[i].item() == 80:
                name = bicycle-human
            elif cls[i].item() == 81:
                name = kickboard-human
            file.write(f"{1 if is_in_red_area else 0} {x} {y} {w} {h} {name}\n")
    print(f"Saved label data for {image_name} to {label_text_path}")
# # 결과 이미지 출력
# plt.figure(figsize=(12,12))
# plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()


xywh_tensor = results[0].boxes.xywh
red_value = [255, 0, 0]  # 실제 마스크의 파란색 값을 기준으로 설정하세요

# 중심 좌표를 추출합니다.
x_center = xywh_tensor[0][0].item()
y_center = xywh_tensor[0][1].item()
width = xywh_tensor[0][2].item()

height = xywh_tensor[0][3].item()

# 중심에서 바닥으로 절반 내린 지점의 y 좌표를 계산합니다.
y_lower_half = y_center + (height / 2)

# 결과 좌표를 반환합니다.
result_coordinates = (x_center, y_lower_half)

# 사람의 좌표를 기반으로 바닥 지점 계산
x_center = xywh_tensor[0][0].item()
y_center = xywh_tensor[0][1].item()
width = xywh_tensor[0][2].item()
height = xywh_tensor[0][3].item()
y_lower_half = y_center + (height / 2)

# 좌표가 정수가 되도록 반올림
x_center_int = int(round(x_center))
y_lower_half_int = int(round(y_lower_half))

# 해당 좌표의 픽셀 값 확인
pixel_value = mask_array[y_lower_half_int, x_center_int]

# 해당 좌표의 픽셀이 파란색인지 확인
is_in_blue_area = np.all(pixel_value == red_value)


# xywh_tensor에서 모든 객체에 대해 반복하여 처리합니다.
for i in range(len(xywh_tensor)):
    # 중심 좌표와 바운딩 박스 크기 추출
    x_center = xywh_tensor[i][0].item()
    y_center = xywh_tensor[i][1].item()
    width = xywh_tensor[i][2].item()
    height = xywh_tensor[i][3].item()

    # 중심에서 바닥으로 절반 내린 지점의 y 좌표를 계산합니다.
    y_lower_half = y_center + (height / 2)

    # 좌표가 정수가 되도록 반올림
    x_center_int = int(round(x_center))
    y_lower_half_int = int(round(y_lower_half))

    # 좌표가 이미지 범위 내에 있는지 확인합니다.
    if 0 <= x_center_int < mask_array.shape[1] and 0 <= y_lower_half_int < mask_array.shape[0]:
        # 해당 좌표의 픽셀 값 확인
        pixel_value = mask_array[y_lower_half_int, x_center_int]

        # 해당 좌표의 픽셀이 빨간색(도로)인지 확인
        is_in_red_area = np.all(pixel_value == red_value)

        print(f"The person at ({x_center}, {y_lower_half})")

        if is_in_red_area:
            print("The person is jaywalking")
        else:
            print("The person is a normal pedestrian.")
    else:
        print(f"The person at ({x_center}, {y_lower_half}) is outside the image boundary.")

