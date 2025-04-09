import cv2
import torch
import yt_dlp

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0, 15, 16]  # 사람, 고양이, 개
class_names = model.names

def get_video_stream(youtube_url):
    ydl_opts = {'format': 'best[ext=mp4]'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict['url']
        fps = info_dict.get('fps', 30)
    return video_url, fps

def process_video(video_url, fps):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("❌ Error: Cannot open video stream.")
        return

    frame_delay = int(1000 / fps)
    frame_count = 0
    boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))  # 속도 향상을 위해 해상도 축소
        frame_count += 1

        # 5프레임마다 객체 탐지 수행
        if frame_count % 5 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame, size=1280)
            boxes = results.xyxy[0].cpu().numpy()

        # 박스 표시
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box[:6]
            if conf < 0.4:
                continue  # 너무 낮은 confidence는 무시

            label = f"{class_names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YouTube Detection (person, cat, dog)", frame)
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=__sBGgK8QJk"
    video_url, fps = get_video_stream(youtube_url)
    fps = 30
    process_video(video_url, fps)
