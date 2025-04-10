import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import LicensePlateRecognizer
import preprocess_input
import detection
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess_input import preprocess_input
import time

# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Nhận diện Biển số và Ký tự")
root.geometry("800x600")

# Tải mô hình YOLO
model = YOLO('D:/mine/HocTap/ThiGiacMay/best.pt')

# Đường dẫn đến mô hình TinyCNN
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   
        self.pool = nn.MaxPool2d(2, 2)                
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_tiny_path = 'D:/mine/HocTap/ThiGiacMay/tinycnn_full_model.pth'  # Đường dẫn đúng đến tệp mô hình .pth của bạn
model_tinycnn = torch.load(model_tiny_path)  # Load mô hình TinyCNN
model_tinycnn.eval()  # Đảm bảo mô hình ở chế độ đánh giá
# Dictionary ánh xạ các nhãn cho ký tự
label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 
              6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 
              12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 
              18: 'K', 19: 'L', 20: 'M', 21: 'N', 22: 'P', 23: 'R', 
              24: 'S', 25: 'T', 26: 'U', 27: 'V', 28: 'X', 29: 'Y', 30: 'Z'}
recognizer = LicensePlateRecognizer.LicensePlateRecognizer(model_tinycnn, label_dict)
def predict(image_path):
    st = time.time()
    cut_plate, bb_img = detection.detect_plate(model, image_path)
    end = time.time()
    start_time = time.time()
    preprocessor = preprocess_input(cut_plate)
    # Gọi phương thức preprocess_image từ đối tượng preprocessor
    img = preprocessor.preprocess_image(preprocessor.image)
    end_time = time.time()
    st_cut = time.time()
    candidates = recognizer.cut_label(img)
    result_text = recognizer.predict_tinycnn(candidates)
    end_cut = time.time()
    print(f"Time for detection plate: {end-st}")
    print(f"Time for preprocessing: {end_time - start_time}")
    print(f"Time for detection char: {end_cut - st_cut}")
    return bb_img, result_text

# Nhãn để hiển thị video hoặc ảnh
video_label = tk.Label(root)
video_label.pack()

# Nhãn để hiển thị văn bản nhận diện
result_text_label = tk.Label(root, text="Nhận diện biển số xe", font=("Arial", 16))
result_text_label.pack()



# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Hàm xử lý video từ camera và nhận diện biển số xe
def process_frame():
    ret, frame = cap.read()

    if ret:
        # Chạy mô hình YOLO trên khung hình hiện tại
        results = model(frame)

        # Lấy ảnh với các hộp giới hạn được vẽ bởi YOLO
        result_img = results[0].plot()  # Vẽ các hộp giới hạn lên ảnh

        # Chuyển đổi ảnh từ BGR (OpenCV) sang RGB để hiển thị trong Tkinter
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Chuyển đổi ảnh sang định dạng mà Tkinter có thể sử dụng
        img_pil = Image.fromarray(result_img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Cập nhật ảnh trên cửa sổ Tkinter
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)

    # Tiếp tục xử lý các khung hình từ camera
    video_label.after(10, process_frame)

# Hàm tải ảnh và thực hiện nhận diện biển số
def load_and_predict_image():
    try:
        # Mở hộp thoại để chọn ảnh
        file_path = filedialog.askopenfilename()

        # Kiểm tra xem người dùng có chọn ảnh không
        if not file_path:
            print("Chưa chọn ảnh")
            return  # Nếu không chọn ảnh thì thoát
        
        # Thực hiện nhận diện biển số xe và lấy văn bản nhận diện
        img, result_text = predict(file_path)

        # Kiểm tra kết quả dự đoán trước khi cập nhật vào Tkinter
        print(f"Văn bản nhận diện: {result_text}")

        # Hiển thị văn bản nhận diện trong cửa sổ Tkinter
        result_text_label.config(text=f"Văn bản nhận diện: {result_text}")

        # Hiển thị ảnh với kết quả nhận diện
        result_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(result_img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Cập nhật ảnh và kết quả trong cửa sổ Tkinter
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")  # In lỗi nếu có vấn đề trong quá trình thực hiện


# Nút để tải ảnh từ máy tính
load_image_button = tk.Button(root, text="Tải ảnh", command=load_and_predict_image)
load_image_button.pack()

# Nút để bắt đầu video từ camera
start_camera_button = tk.Button(root, text="Bắt đầu camera", command=lambda: process_frame())
start_camera_button.pack()


# Khởi động vòng lặp sự kiện Tkinter
root.mainloop()

# Giải phóng camera khi ứng dụng đóng
cap.release()
cv2.destroyAllWindows()