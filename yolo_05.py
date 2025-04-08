import torch
import cv2
from deepface import DeepFace

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("📡 Using device:", device)

# YOLOv5 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # 사람

# 웹캠
cap = cv2.VideoCapture(0)

frame_count = 0
analyze_interval = 10  # 감정 분석을 10프레임마다 수행
cached_emotions = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = model(frame)
    boxes = results.xyxy[0].cpu().numpy()

    face_count = 0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls = map(int, box)
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face_count += 1
        face_id = f"face{face_count}"

        label = f"Face {face_count}"
        if frame_count % analyze_interval == 0:
            try:
                result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                dominant = result[0]['dominant_emotion']
                cached_emotions[face_id] = dominant
            except Exception as e:
                print(f"Error analyzing face: {e}")

        # 캐싱된 감정으로 표시
        if face_id in cached_emotions:
            label += f": {cached_emotions[face_id]}"

        # 시각화
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    frame_count += 1
    cv2.imshow("Real-time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()