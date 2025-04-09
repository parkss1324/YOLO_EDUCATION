import yt_dlp
import cv2
import torch
import os

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("📡 Using device:", device)

# 유튜브 영상 다운로드
def download_video(youtube_url, save_path="downloaded_video.mp4"):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': save_path,
        'merge_output_format': 'mp4',
        'quiet': False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return save_path

# YOLO 감지 및 표시
def process_video(video_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.4  # 감지 임계값
    model.classes = [0, 2, 5, 7, 9]  # 추론할 클래스 제한

    # 감지할 클래스명 매핑
    target_classes = {
        0: 'person',
        2: 'car',
        5: 'bus',
        7: 'truck',
        9: 'traffic light'
    }

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("🎬 영상 끝!")
            break

        frame_count += 1
        if frame_count % 2 != 0:
            cv2.imshow("YOLOv5 - Local Video", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue

        resized_frame = cv2.resize(frame, (640, 360))  # 원본보다 작게 리사이즈
        results = model(resized_frame)

        boxes = results.xyxy[0].cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)
            if cls not in target_classes:
                continue

            label = f"{target_classes[cls]} ({conf:.2f})"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("YOLOv5 - Local Video", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 메인 실행
if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=RubZi99YX14"
    local_file = "downloaded_video.mp4"

    if not os.path.exists(local_file):
        print("📥 영상 다운로드 중...")
        download_video(url, local_file)

    print("🚀 영상 처리 시작")
    process_video(local_file)
