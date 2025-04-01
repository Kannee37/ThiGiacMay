# import cv2
# import numpy as np
# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# def show_image(title, img):
#     plt.figure(figsize=(8, 8))
#     if len(img.shape) == 2:
#         plt.imshow(img, cmap='gray')
#     else:
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(title)
#     plt.axis('off')
#     plt.show()


# def preprocess_image(image_path):
#     # Đọc ảnh
#     img = cv2.imread(image_path)
#     show_image("Original Image", img)
    
#     # Bước 1: Khử nhiễu (Noise Reduction)
#     denoised_img = cv2.GaussianBlur(img, (5, 5), 0)  # Kernel lớn để khử noise hiệu quả
#     show_image("Denoised Image", denoised_img)
    
#     # Bước 2: Chuyển ảnh sang ảnh xám (Grayscale)
#     gray_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
#     show_image("Grayscale Image", gray_img)
    
#     # Bước 3: Làm nét ảnh (Sharpening)
#     kernel = np.array([[0, -1, 0], [-1, 4.8, -1], [0, -1, 0]])  # Kernel mạnh để làm nét
#     sharpened_img = cv2.filter2D(gray_img, -1, kernel)
#     show_image("Sharpened Image", sharpened_img)
    
#     # Bước 4: Nhị phân hóa ảnh (Binary Thresholding)
#     _, binary_img = cv2.threshold(sharpened_img, 150, 255, cv2.THRESH_BINARY)
#     show_image("Binary Image", binary_img)

#     # Chuyển lại về ảnh màu (3 kênh) để phù hợp đầu vào của YOLOv11
#     final_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
#     return final_img

# def load_model(model_path):
#     return YOLO(model_path)

# def detect_plate(model, img):
#     results = model.predict(img, save=True, save_txt=True, conf=0.25)
#     result_img = results[0].plot()
#     show_image("Detection Result", result_img)

# if __name__ == "__main__":
#     # Đường dẫn đến mô hình YOLO đã huấn luyện
#     model_path = "C:/Users/ADMIN/Documents/GitHub/ThiGiacMay/runs/detect/train/weights/best.pt"
#     model = load_model(model_path)

#     # Đường dẫn đến ảnh cần nhận diện
#     image_path = "C:/Users/ADMIN/Documents/GitHub/ThiGiacMay/xemay.jpg"
    
#     # Tiền xử lý ảnh
#     processed_img = preprocess_image(image_path)
    
#     # Nhận diện biển số xe bằng YOLO
#     detect_plate(model, processed_img)

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

def show_image(title, img):
    plt.figure(figsize=(8, 8))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def preprocess_image(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)
    show_image("Original Image", img)
    
    # Bước 1: Khử nhiễu (Noise Reduction)
    denoised_img = cv2.GaussianBlur(img, (5, 5), 0)  # Kernel lớn để khử noise hiệu quả
    show_image("Denoised Image", denoised_img)
    
    # Bước 2: Chuyển ảnh sang ảnh xám (Grayscale)
    gray_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
    show_image("Grayscale Image", gray_img)
    
    # Bước 3: Làm nét ảnh (Sharpening)
    kernel = np.array([[0, -1, 0], [-1, 4.8, -1], [0, -1, 0]])  # Kernel mạnh để làm nét
    sharpened_img = cv2.filter2D(gray_img, -1, kernel)
    show_image("Sharpened Image", sharpened_img)
    
    # Bước 4: Nhị phân hóa ảnh (Binary Thresholding)
    _, binary_img = cv2.threshold(sharpened_img, 150, 255, cv2.THRESH_BINARY)
    show_image("Binary Image", binary_img)

    # Chuyển lại về ảnh màu (3 kênh) để phù hợp đầu vào của YOLOv11
    final_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    return final_img

def load_model(model_path):
    return YOLO(model_path)

def detect_plate(model, img):
    results = model.predict(img, save=True, save_txt=True, conf=0.25)
    
    # Lấy các bounding box của các đối tượng được phát hiện
    boxes = results[0].boxes.xyxy.numpy()  # Bounding boxes theo dạng [x1, y1, x2, y2]
    
    if len(boxes) > 0:
        cropped_plates = []  # Danh sách để lưu các biển số đã cắt ra
        closing_plates = []  # Danh sách để lưu các biển số sau khi áp dụng Closing
        
        # Lặp qua tất cả các bounding box và cắt biển số ra
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Lấy tọa độ cắt
            cropped_plate = img[y1:y2, x1:x2]  # Cắt biển số
            
            # Áp dụng phép Closing lên biển số cắt được
            closing_img = apply_closing(cropped_plate)
            
            # Hiển thị kết quả từng biển số
            show_image(f"Cropped Plate {idx + 1}", cropped_plate)
            show_image(f"Plate {idx + 1} after Closing", closing_img)
            
            # Thêm vào danh sách biển số đã cắt và đã xử lý
            cropped_plates.append(cropped_plate)
            closing_plates.append(closing_img)
        
        return cropped_plates, closing_plates

def apply_closing(img):
    # Áp dụng Closing: Dilation rồi Erosion
    kernel = np.ones((1,1), np.uint8)  # Kernel cho Closing
    closing_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing_img

if __name__ == "__main__":
    # Đường dẫn đến mô hình YOLO đã huấn luyện
    model_path = "C:/Users/ADMIN/Documents/GitHub/ThiGiacMay/runs/detect/train/weights/best.pt"
    model = load_model(model_path)

    # Đường dẫn đến ảnh cần nhận diện
    image_path = "C:/Users/ADMIN/Documents/GitHub/ThiGiacMay/xemay.jpg"
    
    # Tiền xử lý ảnh
    processed_img = preprocess_image(image_path)
    
    # Nhận diện biển số xe bằng YOLO và cắt từng biển số ra
    cropped_plates, closing_plates = detect_plate(model, processed_img)
