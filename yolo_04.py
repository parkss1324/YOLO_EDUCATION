import torch
import cv2
from deepface import DeepFace
from collections import defaultdict

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("📡 Using device:", device)

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # 사람

# 이미지 불러오기
img_path = "/Users/parksungsu/Documents/python_opencv/yolov5/data/images/zidane.jpg"
img = cv2.imread(img_path)

# YOLO 추론
results = model(img)
boxes = results.xyxy[0].cpu().numpy()

# 감정 분포 누적용
emotion_totals = defaultdict(float)
face_count = 0

for box in boxes:
    x1, y1, x2, y2, conf, cls = map(int, box)
    face = img[y1:y2, x1:x2]

    if face.size == 0:
        continue

    try:
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        emotions = result[0]['emotion']
        dominant = result[0]['dominant_emotion']
        face_count += 1

        print(f"\n🧑 Face {face_count} - Dominant Emotion: {dominant}")
        for emo, score in emotions.items():
            emotion_totals[emo] += score
            print(f"{emo}: {score:.2f}%")

        # 얼굴 영역 표시 + Face 번호와 감정 텍스트 추가
        label = f"Face {face_count}: {dominant}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    except Exception as e:
        print(f"Error analyzing face: {e}")

# 결과 이미지 출력
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
