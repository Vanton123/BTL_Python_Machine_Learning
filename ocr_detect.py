
from paddleocr import PaddleOCR
import cv2
import os
import numpy as np

def enhance_image_for_ocr(image):
    # 1. Resize trước để xử lý trên ảnh lớn hơn (giúp giữ nét khi threshold)
    # Scale x3 để chữ to rõ ràng hơn
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # 2. Chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Nhị phân hóa (Threshold) để tách hẳn chữ ra khỏi nền
    # Dùng Otsu's threshold để tự động tìm ngưỡng tối ưu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. KIỂM TRA VÀ ĐẢO NGƯỢC MÀU (QUAN TRỌNG NHẤT)
    # PaddleOCR thích chữ ĐEN nền TRẮNG.
    # Đếm số pixel trắng. Nếu pixel trắng nhiều hơn đen => Nền trắng chữ đen (ok).
    # Nếu pixel trắng ít hơn đen => Nền đen chữ trắng (cần đảo ngược).
    
    white_pixels = np.count_nonzero(binary == 255)
    total_pixels = binary.size
    
    # Nếu số pixel trắng chiếm < 50% (tức là nền đang đen/tối), ta đảo ngược lại
    if white_pixels < (total_pixels * 0.5):
        binary = cv2.bitwise_not(binary)

    # 5. Giãn nở nhẹ (Dilation) để làm đậm nét chữ nếu chữ bị đứt nét (tùy chọn)
    # kernel = np.ones((2,2), np.uint8)
    # binary = cv2.dilate(binary, kernel, iterations=1)

    return binary

def detect_license_plate_optimized(image_path):
    # Khởi tạo PaddleOCR (tắt cảnh báo debug)
    ocr = PaddleOCR(lang='en')

    img = cv2.imread(image_path)
    if img is None:
        print(f"Không tìm thấy ảnh tại: {image_path}")
        return

    # Gọi hàm xử lý ảnh mới
    processed_img = enhance_image_for_ocr(img)
    processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    
    # Debug: Lưu hoặc hiện ảnh đã xử lý để xem mắt thường có đọc được không
    cv2.imshow("Debug Processed", processed_img) 
    cv2.waitKey(0) 

    # PaddleOCR nhận diện
    # det=False: Vì ảnh đã là biển số cắt sẵn, ta bỏ qua bước phát hiện vùng (detection)
    # để tập trung vào nhận diện ký tự (recognition) sẽ chính xác hơn.
    # result=ocr.predict(image_path)
    result = ocr.predict(processed_img_bgr)

    # Xử lý kết quả
    all_texts = []
    all_scores = []

    for line in result:
        texts = line.get('rec_texts', [])
        scores = line.get('rec_scores', [])

        for text, score in zip(texts, scores):
            cleaned_text = text.strip()
            if len(cleaned_text) > 0:
                all_texts.append(cleaned_text)
                all_scores.append(score)
                print(f" '{text}' (confidence: {score:.4f})")

    license_plate = "".join(all_texts)
    avg_confidence = sum(all_scores) / len(all_scores)

    print(f"Biển số: {license_plate}")
    print(f"Độ tin cậy: {avg_confidence:.4f}")

if __name__ == "__main__":    
    image_path = r'C:\Users\Acer\OneDrive - ut.edu.vn\Documents\MachineLearningPython\detectImage\bien_so_New_3.jpg'
    detect_license_plate_optimized(image_path)
