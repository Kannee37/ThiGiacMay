from ultralytics import YOLO  # Import YOLO11 từ ultralytics
import cv2


# Đường dẫn đến mô hình YOLO11n đã huấn luyện (ví dụ: YOLO11n.pt)
model_path = "C:/Users/ADMIN/Documents/GitHub/ThiGiacMay/runs/detect/train/weights/best.pt"

# Load mô hình YOLO11n đã huấn luyện
model = YOLO(model_path)  # Sử dụng thư viện YOLO11 từ ultralytics

# Mở camera (0 là camera mặc định)
cap = cv2.VideoCapture(0)

#NHẬN DIỆN CAMERA
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán bằng mô hình YOLO11
    results = model(frame)

    # Lấy các thông tin nhận diện từ kết quả dự đoán (YOLO11)
    detections = results[0].boxes.data  # Lấy kết quả nhận diện từ YOLO11n

    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        confidence = round(conf * 100, 2)
        label = model.names[int(cls)]

        # Vẽ khung bao và nhãn
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị khung hình nhận diện
    cv2.imshow('YOLO11 Detection', frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổq
cap.release()
cv2.destroyAllWindows()
