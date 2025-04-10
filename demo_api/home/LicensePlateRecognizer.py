import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage.filters import threshold_local
from skimage import measure
import torch
import torch.nn

class LicensePlateRecognizer:
    def __init__(self, model, label_dict, target_size=(32, 32)):
        self.model = model
        self.label_dict = label_dict
        self.target_size = target_size

    def cropped_bolder(self, img, t):
        h, w = img.shape[:2]
        top_px = int(h * t)
        bottom_px = int(h * t)
        left_px = int(w * t)
        right_px = int(w * t)
        cropped = img[top_px:h - bottom_px, left_px:w - right_px]
        return cropped

    def convert_white_to_gray(self, img):
        # Đảm bảo ảnh có 1 kênh (grayscale)
        if len(img.shape) == 3:
            # Nếu ảnh là ảnh RGB, chuyển nó sang grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Thay thế pixel trắng (255) thành màu xám (128)
        # Tạo một bản sao của ảnh
        modified_img = np.copy(img)

        # Thay pixel trắng (255) thành xám (128)
        modified_img[modified_img == 255] = 128

        return modified_img

    def add_padding(self, image):
        h, w = image.shape[:2]
        target_h, target_w = self.target_size

        # Tính toán padding cần thiết
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left

        # Thêm padding vào ảnh
        padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        return padded_image

    def prepro_img_forcut(self, img):
        img = self.cropped_bolder(img, 0.17)
        V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
        thresh = cv2.resize(thresh, (400, int(thresh.shape[0] * 400 / thresh.shape[1])))
        thresh = cv2.medianBlur(thresh, 5)
        return thresh

    def cut_label(self, img):
        candidates = []  # Chứa ảnh ký tự được cắt ra

        # xử lý ảnh
        thresh = self.prepro_img_forcut(img)
        
        # Làm sạch ảnh với morphological operations (ví dụ: Closing)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Phân tích thành phần liên kết, hàm measure.label()
        labels = measure.label(thresh, connectivity=2, background=0)

        bounding_boxes = []  # Danh sách các bounding box

        # Lặp qua các thành phần liên kết (labels)
        for label in np.unique(labels):
            if label == 0:  # Nếu là nền, bỏ qua
                continue

            # Tạo mặt nạ để lưu trữ vị trí của các ký tự
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            # Tìm các đường viền (contours) trong mặt nạ
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)  # Chọn đường viền có diện tích lớn nhất
                (x, y, w, h) = cv2.boundingRect(contour)  # Lấy bounding box của ký tự

                # Các quy tắc để xác định ký tự hợp lệ
                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(img.shape[0])

                # Kiểm tra các quy tắc (tỷ lệ, độ đặc, chiều cao)
                if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                    # Cắt ký tự từ mặt nạ
                    candidate = np.array(mask[y:y + h, x:x + w])
                    candidate_resized = cv2.resize(candidate, (28, 28), cv2.INTER_AREA)
                    candidate_resized = candidate_resized.reshape((28, 28, 1))  # Đảm bảo ảnh có đúng 1 kênh
                    candidates.append((candidate_resized, (x, y)))  # Thêm (ảnh cắt, vị trí x, y)

                    # Thêm bounding box vào danh sách
                    bounding_boxes.append((x, y, w, h))

        # Chia ảnh thành 3 phần theo chiều dọc (top, middle, bottom)
        height = img.shape[0]
        top_threshold = height // 3
        bottom_threshold = 2 * height // 3

        top_row_boxes = [box for box in bounding_boxes if box[1] < top_threshold]
        middle_row_boxes = [box for box in bounding_boxes if top_threshold <= box[1] < bottom_threshold]
        bottom_row_boxes = [box for box in bounding_boxes if box[1] >= bottom_threshold]

        # Sắp xếp các bounding box trong mỗi phần theo tọa độ x (từ trái sang phải)
        top_row_boxes = sorted(top_row_boxes, key=lambda box: box[0])
        middle_row_boxes = sorted(middle_row_boxes, key=lambda box: box[0])
        bottom_row_boxes = sorted(bottom_row_boxes, key=lambda box: box[0])

        # Kết hợp các bounding box đã sắp xếp
        sorted_boxes = top_row_boxes + middle_row_boxes + bottom_row_boxes

        sorted_candidates = []
        for box in sorted_boxes:
            for i, (candidate, (x, y)) in enumerate(candidates):
                if (x, y) == (box[0], box[1]):
                    sorted_candidates.append(candidates[i])

        # Vẽ bounding box và đánh số thứ tự
        img_with_boxes = thresh.copy()
        for idx, (x, y, w, h) in enumerate(sorted_boxes):
            # Vẽ bounding box
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Đánh số thứ tự cho từng bounding box
            cv2.putText(img_with_boxes, str(idx + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # cv2_imshow(img_with_boxes)

        return sorted_candidates

    def predict_cnn(self, candidates):
        end = ''
        for i, (char_image, _) in enumerate(candidates):
            # Dự đoán
            pre = self.model_cnn_predict_char(char_image)
            end += pre
            print(pre)
        return end

    def predict_tinycnn(self, candidates):
        end = ''
        for i, (char_image, _) in enumerate(candidates):
            # Tiền xử lý và dự đoán
            pre = self.model_tinycnn_predict_char(char_image)
            end += pre
            print(pre)
        return end
        
    def preprocess_image(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32))  # Đổi kích thước ảnh
        img = img / 255.0  # Chuẩn hóa giá trị pixel
        if len(img.shape) == 2:  # Nếu ảnh là 2 chiều (height, width)
            img = np.expand_dims(img, axis=0)  
        return img

    def model_cnn_predict_char(self, img):
        img = self.prepro_image_for_predict(img)
        pre = self.model.predict(np.expand_dims(img, axis=0))  # Dự đoán với ảnh đã thêm padding
        pre = np.argmax(pre, axis=1)[0]  # Lấy nhãn có xác suất cao nhất
        result = self.label_dict[pre]
        return result

    def prepro_image_for_predict(self, img):
        padded_image = self.add_padding(img)
        gray_w = self.convert_white_to_gray(padded_image)
        cv2.imshow(gray_w)
        return gray_w

    def model_tinycnn_predict_char(self, img):
        img = self.prepro_image_for_predict(img)
        img = self.preprocess_image(img)
        self.model.eval()  # Chuyển mô hình về chế độ đánh giá
        img = torch.tensor(img, dtype=torch.float32)  # Chuyển thành tensor 
        with torch.no_grad():  # Tắt gradient để tiết kiệm bộ nhớ
            output = self.model(img)
            _, predicted = torch.max(output, 1)  # Lấy lớp có xác suất cao nhất
        result = self.label_dict[predicted.item()]  # Trả về nhãn dự đoán
        return str(result)


# # Đường dẫn đến mô hình .h5 và bảng ánh xạ nhãn
# model = 'D:/mine/HocTap/ThiGiacMay/license_plate_recognition_model.h5'  # Đường dẫn đến tệp mô hình .h5 của bạn
# label = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 
#               6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 
#               12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 
#               18: 'K', 19: 'L', 20: 'M', 21: 'N', 22: 'P', 23: 'R', 
#               24: 'S', 25: 'T', 26: 'U', 27: 'V', 28: 'X', 29: 'Y', 30: 'Z'}


# # Khởi tạo và gọi mô hình
# def run_license_plate_recognition(image_path, model_path = model, label_dict = label):
#     recognizer = LicensePlateRecognizer(model_path, label_dict)
#     img = cv2.imread(image_path)
#     candidates = recognizer.cut_label(img)
#     result = recognizer.predict(candidates)
#     return result