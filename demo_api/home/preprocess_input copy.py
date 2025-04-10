import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np

class preprocess_input:
    def __init__ (self, image):
        self.image = image

    def check_sharpness(self, image):
        # Tính gradient của ảnh sử dụng Sobel filter
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)  # Tính độ lớn gradient
        
        # Tính độ tương phản (biến thiên gradient)
        sharpness = np.mean(magnitude)
        return sharpness

    def sharpen_image(self, image):
        # Bộ lọc làm nét (sharpening kernel)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened_img = cv2.filter2D(image, -1, kernel)
        return sharpened_img
    
    def check_noise(self, image):
        # Sử dụng Laplacian để phát hiện nhiễu
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = laplacian.var()  # Đo độ biến thiên của Laplacian
        return variance

    def denoise_gaussian(self, image):
        # Áp dụng bộ lọc Gaussian để loại bỏ nhiễu
        denoised_img = cv2.GaussianBlur(image, (5, 5), 0)
        return denoised_img
    
    # kiểm tra độ xoay
    def angle_of_rotation(self, image):
        # Bước 1: Đọc ảnh và chuyển sang ảnh xám
        img = image.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Bước 2: Áp dụng edge detection (Canny)
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        # Bước 3: Áp dụng phép biến đổi Hough để phát hiện đường thẳng
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 90)  # ngưỡng: 100
        # Kiểm tra nếu có các đường thẳng phát hiện được
        if lines is not None:
            # Lấy góc của các đường thẳng
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)  # Chuyển đổi từ radians sang độ
                angles.append(angle)
                # Điều chỉnh góc nếu nó lớn hơn 90 độ
                if angle > 90:
                    angle -= 180
                angles.append(angle)
            # Tính góc trung bình của tất cả các đường thẳng
            mean_angle = np.mean(angles)
            return mean_angle
        else:
            print("Không phát hiện được đường thẳng.")
            return None

    def rotate_image_to_horizontal(self, image, angle = 0):
        # Lấy kích thước của ảnh
        (h, w) = image.shape[:2]
        # Tính toán trung tâm ảnh
        center = (w // 2, h // 2)
        # Tạo ma trận xoay
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        # Xoay ảnh với ma trận và giữ lại toàn bộ ảnh sau khi xoay
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated_image

    def apply_clahe(self, image):
        # Chuyển ảnh sang xám nếu là ảnh màu
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Khởi tạo CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Bạn có thể điều chỉnh clipLimit và tileGridSize
        clahe_img = clahe.apply(gray)
        return clahe_img
    
    def preprocess_image(self, image):
        sharpness_value = self.check_sharpness(image)
        noise_value = self.check_noise(image)
        angle = self.angle_of_rotation(image)
        
        if sharpness_value < 70:
            print("Ảnh mờ, cần làm sắc nét.")
            image = self.sharpen_image(image)
        if noise_value > 1000:
            print("Ảnh có nhiễu, cần giảm nhiễu.")
            image = self.denoise_gaussian(image)
        if angle != None:
            print("Ảnh cần xoay")
            image = self.rotate_image_to_horizontal(image, 90-angle)
        return image