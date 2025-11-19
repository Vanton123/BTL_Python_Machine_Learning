# from paddleocr import PaddleOCR 
from ultralytics import YOLO
from PIL import Image
import cv2
import os

# Load model đã train
model_path = r'C:\Users\Acer\OneDrive - ut.edu.vn\Documents\MachineLearningPython\runs\detect\train2\weights\best.pt'
model = YOLO(model_path)

# tên file ảnh
image_path = r'C:\Users\Acer\OneDrive - ut.edu.vn\Documents\MachineLearningPython\fileAnh\z7207866535006_7d4887309446deef051ab292db417596.jpg'

# folder lưu vung biển số bị cắt
folderdetect=r'C:\Users\Acer\OneDrive - ut.edu.vn\Documents\MachineLearningPython\detectImage'

# Hiển thị kết quả nhận diện
res=model(image_path)
for r in res:
    image_arr=r.plot()
    image=Image.fromarray(image_arr[...,::-1])
    image.show()
    image.save("kq.jpg")


results = model.predict(image_path,conf=0.5)
res = results[0]
print("Phát hiện xong")

# Đoc ảnh
img=cv2.imread(image_path)

# Lưu vùng biển số
for i, box in enumerate(res.boxes.xyxy):
    x1,y1,x2,y2=map(int,box)

    crop=img[y1:y2,x1:x2]
    crop_path=os.path.join(folderdetect,f"bien_so_New_3.jpg")
    cv2.imwrite(crop_path,crop)
    print("Da luu vung bien so")
