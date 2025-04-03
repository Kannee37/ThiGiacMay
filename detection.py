import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import time  

# Hàm hiển thị ảnh
def show_image(title, img):
    plt.figure(figsize=(8, 8))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

#xử lý ảnh
def preprocess_noisy_image(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)
    show_image("Original Image", img)
    
    # Bước 1: Khử nhiễu (Noise Reduction)
    denoised_img = cv2.GaussianBlur(img, (5, 5), 0)  # Kernel lớn để khử noise hiệu quả
    # denoised_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    show_image("Denoised Image", denoised_img)
    
    # kernel = np.array([[0, -1, 0], [-1, 4.8, -1], [0, -1, 0]])  # Kernel mạnh để làm nét
    # sharpened_img = cv2.filter2D(denoised_img, -1, kernel)
    # show_image("Sharpened Image", sharpened_img)
    
    # Bước 2: Chuyển ảnh sang ảnh xám (Grayscale)
    gray_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
    show_image("Grayscale Image", gray_img)
    
    # Bước 3: Làm nét ảnh (Sharpening)
    kernel = np.array([[0, -1, 0], [-1, 4.8, -1], [0, -1, 0]])  # Kernel mạnh để làm nét
    sharpened_img = cv2.filter2D(gray_img, -1, kernel)
    show_image("Sharpened Image", sharpened_img)
    
    # Bước 4: Nhị phân hóa ảnh bằng ngưỡng động
    binary_img = cv2.adaptiveThreshold(gray_img, 255, 
                                      cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 
                                      11, 2)  # Tham số 11: kích thước của vùng ô vuông (block size), 2: giá trị điều chỉnh
    show_image("Binary Image", binary_img)

    # Chuyển lại về ảnh màu (3 kênh) để phù hợp đầu vào của YOLOv11
    final_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    return final_img

# Hàm tải mô hình YOLO
def load_model(model_path):
    return YOLO(model_path)

# Hàm cắt ảnh từ thư mục (nhiều ảnh)
def cut_images_from_directory(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục lưu ảnh nếu chưa có

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Kiểm tra định dạng ảnh
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # Dự đoán với mô hình YOLO
            results = model(img)

            # Kiểm tra số lượng bounding boxes được phát hiện
            detected_count = len(results[0].boxes)
            if detected_count == 0:
                print(f"Không phát hiện biển số trong ảnh: {filename}")
            else:
                print(f"Phát hiện {detected_count} biển số trong ảnh: {filename}")

                # Lặp qua tất cả các bounding box và cắt ảnh biển số
                for i, result in enumerate(results[0].boxes):
                    coords = result.xyxy.tolist()[0]  # Convert tensor to list and access the first element
                    xmin, ymin, xmax, ymax = map(int, coords)

                    # Cắt ảnh biển số
                    cropped_img = img[ymin:ymax, xmin:xmax]

                    # Thêm dấu thời gian vào tên file ảnh cắt
                    timestamp = int(time.time())  # Dấu thời gian để tạo tên file duy nhất
                    output_filename = f"{os.path.splitext(filename)[0]}_cropped_{i}_{timestamp}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, cropped_img)
                    print(f"Đã lưu ảnh biển số xe tại: {output_path}")
    print("ĐÃ CẮT ẢNH XONG")

# Hàm cắt ảnh đơn (một ảnh)
def cut_single_image(model, img_path, output_dir):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {img_path}")
        return [], None

    # Dự đoán với mô hình YOLO
    results = model.predict(img, save=True, save_txt=True, conf=0.25)

    # Sử dụng phương thức plot() để vẽ bounding box và nhãn
    result_img = results[0].plot()  # Tự động vẽ bounding box và nhãn
    
    # Hiển thị kết quả nhận diện
    show_image("Detection Result", result_img)

    # Lấy các bounding box và nhãn của các đối tượng được phát hiện
    boxes = results[0].boxes.xyxy.numpy()  # Bounding boxes theo dạng [x1, y1, x2, y2]
    
    cropped_plates = []  # Danh sách lưu các biển số đã cắt ra

    # Lặp qua tất cả các bounding box và cắt biển số ra
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  # Lấy tọa độ cắt
        cropped_plate = img[y1:y2, x1:x2]  # Cắt biển số
        
        # Thêm dấu thời gian vào tên file ảnh cắt
        timestamp = int(time.time())  # Dấu thời gian để tạo tên file duy nhất
        output_filename = f"cropped_plate_{idx + 1}_{timestamp}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, cropped_plate)
        
        # Hiển thị kết quả từng biển số
        show_image(f"Cropped Plate {idx + 1}", cropped_plate)
        
        # Thêm vào danh sách biển số đã cắt
        cropped_plates.append(cropped_plate)
    
    # In ra thông báo lưu thành công
    print(f"Đã lưu tất cả biển số đã cắt vào thư mục: {output_dir}")
    
    return cropped_plates, result_img

# Hàm chính để nhận diện biển số xe
def detect_plate(model, img_path, output_dir, input_dir=None):
    # Nếu có thư mục input_dir, xử lý tất cả các ảnh trong thư mục đó
    if input_dir:
        cut_images_from_directory(model, input_dir, output_dir)  # Xử lý tất cả ảnh trong thư mục
        return

    # Nếu chỉ có một ảnh đầu vào, xử lý ảnh đó
    return cut_single_image(model, img_path, output_dir)

if __name__ == "__main__":
    # Đường dẫn đến mô hình YOLO đã huấn luyện
    model_path = "C:/Users/ADMIN/Documents/GitHub/ThiGiacMay/runs/detect/train/weights/best.pt"
    model = load_model(model_path)

    # Đường dẫn đến ảnh cần nhận diện
    image_path = "C:/Users/ADMIN/Documents/GitHub/ThiGiacMay/xe.png"
    
    # Tiền xử lý ảnh
    # processed_img = preprocess_noisy_image(image_path)

    # Đường dẫn thư mục để lưu ảnh cắt
    output_dir = "C:/Users/ADMIN/Documents/GitHub/ThiGiacMay/cropped_images"  # Thư mục đầu ra
    
    # Nhận diện biển số xe bằng YOLO và cắt từng biển số ra
    cropped_plates, closing_plates = detect_plate(model, image_path, output_dir)