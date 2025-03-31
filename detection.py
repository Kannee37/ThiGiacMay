import cv2
import torch
import numpy as np
## Vẫn còn lỗi vài chỗ
def load_model(model_path):
    """Tải mô hình YOLOv11 để nhận diện biển số xe."""
    model = torch.hub.load('ultralytics/yolov11', 'custom', path=model_path, force_reload=True)
    return model
def detect_license_plate(model, image_path, conf_threshold=0.5):
    """Nhận diện biển số xe trong ảnh bằng mô hình YOLOv11."""
    # Đọc ảnh đầu vào
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    # Chạy mô hình YOLO để phát hiện đối tượng
    results = model(image)
    detections = results.xyxy[0]  # Lấy danh sách các đối tượng phát hiện được
    
    license_plates = []  # Danh sách biển số xe được phát hiện
    for *xyxy, conf, cls in detections:
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, xyxy)  # Lấy tọa độ hộp giới hạn
            plate_img = image[y1:y2, x1:x2]  # Cắt vùng chứa biển số
            license_plates.append((plate_img, (x1, y1, x2, y2)))
    
    return license_plates

def save_license_plates(plates, output_dir):
    """Lưu ảnh biển số đã cắt vào thư mục chỉ định."""
    for i, (plate, _) in enumerate(plates):
        cv2.imwrite(f"{output_dir}/plate_{i}.jpg", plate)

if __name__ == "__main__":
    model_path = "best.pt"  # Đường dẫn đến mô hình YOLOv11 đã huấn luyện
    image_path = "test.jpg"  # Đường dẫn ảnh đầu vào
    output_dir = "./cropped_plates"  # Thư mục lưu biển số đã cắt
    
    model = load_model(model_path)  # Tải mô hình
    plates = detect_license_plate(model, image_path)  # Phát hiện biển số
    save_license_plates(plates, output_dir)  # Lưu ảnh biển số
    
    print(f"Đã phát hiện và lưu {len(plates)} biển số xe.")