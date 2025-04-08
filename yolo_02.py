from deepface import DeepFace
import cv2

img_path = "/Users/parksungsu/Documents/python_opencv/yolov5/data/images/zidane.jpg"
img = cv2.imread(img_path)

result = DeepFace.analyze(img_path=img, actions=['emotion'])
print(result[0]['dominant_emotion'])