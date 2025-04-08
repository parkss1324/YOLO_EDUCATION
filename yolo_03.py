import torch
import cv2
from deepface import DeepFace

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("📡 Using device:", device)

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # 클래스 0은 사람

# 이미지 불러오기
img_path = "/Users/parksungsu/Documents/python_opencv/yolov5/data/images/zidane.jpg"
img = cv2.imread(img_path)

# YOLO 추론
results = model(img)

# 바운딩 박스 정보 가져오기
boxes = results.xyxy[0].cpu().numpy()

for box in boxes:
    x1, y1, x2, y2, conf, cls = map(int, box)
    face = img[y1:y2, x1:x2]

    # 얼굴이 유효한 경우만 분석
    if face.size == 0:
        continue

    try:
        # DeepFace 감정 분석
        result = DeepFace.analyze(img_path=face, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        print(f"Detected emotion: {emotion}")

        # 결과 출력
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    except Exception as e:
        print(f"Error analyzing face: {e}")

# 결과 이미지 출력
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
