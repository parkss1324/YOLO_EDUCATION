import torch
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

# 1. 디바이스 설정: MPS (Mac GPU) 또는 CPU fallback
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("📡 Using device:", device)

# 2. 모델 로드 (YOLOv5s - 작은 모델)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)

# 3. 이미지 불러오기
img_path = '/Users/parksungsu/Documents/python_opencv/yolov5/data/images/bus.jpg'  # 예제 이미지 경로 (YOLOv5 기본 포함됨)
img = cv2.imread(img_path)[..., ::-1]  # BGR → RGB

# 4. 추론
results = model(img)

# 5. 결과 표시
results.print()
results.show()  # matplotlib 창으로 이미지 띄움

# 6. 결과 저장 (선택)
results.save(save_dir='results')  # 'results/' 폴더에 저장됨