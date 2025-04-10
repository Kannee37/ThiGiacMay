import cv2
from ultralytics import YOLO
import LicensePlateRecognizer
from preprocess_input import preprocess_input
import detection
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib as plt

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


img_path = 'D:/mine/HocTap/ThiGiacMay/GreenParking/0000_00532_b.jpg'
start_time = time.time()
img, re = predict(img_path)
end_time = time.time()
print(re)
print(f"Thời gian nhận diện: {end_time - start_time}")
cv2.imshow("Detection Result", img)  # Hiển thị ảnh với bounding box
cv2.waitKey(0)
cv2.destroyAllWindows()

