import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Định nghĩa đường dẫn đến bộ dữ liệu
data_dir = "D:/mine/HocTap/ThiGiacMay/char_bien_so"  # Đường dẫn đến thư mục chứa dữ liệu

# Hàm tiền xử lý ảnh: đổi ảnh thành grayscale, chuẩn hóa và thay đổi kích thước
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))  # Đổi kích thước ảnh
    img = img / 255.0  # Chuẩn hóa giá trị pixel
    img = np.expand_dims(img, axis=-1)  # Thêm chiều kênh màu (1 kênh cho ảnh grayscale)
    return img

# Hàm chuẩn bị dữ liệu
def prepare_data(data_dir):
    images = []
    labels = []
    label_dict = {}  # Lưu nhãn tương ứng với các thư mục
    label_id = 0  # Khởi tạo nhãn bắt đầu từ 0

    # Đọc ảnh từ các thư mục trong bộ dữ liệu
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            label_dict[label_id] = folder  # Gán nhãn cho từng thư mục
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                img = preprocess_image(image_path)
                images.append(img)
                labels.append(label_id)
            label_id += 1  # Tăng nhãn cho thư mục tiếp theo

    # Chuyển danh sách ảnh và nhãn thành numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, label_dict

# Chuẩn bị dữ liệu
images, labels, label_dict = prepare_data(data_dir)

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Định nghĩa mô hình CNN
model = models.Sequential([
    layers.InputLayer(input_shape=(64, 64, 1)),  # Ảnh grayscale 64x64 với 1 kênh
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_dict), activation='softmax')  # Số lớp = số ký tự
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Lưu mô hình
model.save('license_plate_recognition_model.h5')

# Dự đoán từ ảnh mới
def predict_from_image(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Thêm chiều batch
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction, axis=1)[0]  # Dự đoán nhãn
    return label_dict[predicted_label]  # Trả về ký tự dự đoán

# Ví dụ sử dụng dự đoán
image_path = "D:\mine\HocTap\ThiGiacMay\image copy 2.png"
predicted_char = predict_from_image(image_path)
print(f"Predicted Character: {predicted_char}")