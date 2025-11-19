# from ultralytics import YOLO

# model=YOLO('yolo11s.pt')

# if __name__=='__main__':
#     result=model.train(data=r'C:\Users\Acer\OneDrive - ut.edu.vn\Documents\MachineLearningPython\vantonBienSoDetect.v2i.yolov11\data.yaml',epochs=50,batch=8,device='cuda')
from ultralytics import YOLO

# Load lại lần train cũ
model = YOLO(r'C:\Users\Acer\OneDrive - ut.edu.vn\Documents\MachineLearningPython\runs\detect\train2\weights\last.pt')

if __name__ == '__main__':
    result = model.train(
        data=r'C:\Users\Acer\OneDrive - ut.edu.vn\Documents\MachineLearningPython\vantonBienSoDetect.v2i.yolov11\data.yaml',
        epochs=50,       
        batch=2,         
        imgsz=640,       
        workers=0,      
        device='cuda',
        resume=True     
    )
