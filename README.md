# Nhận Diện Biển Số Xe
## Giới thiệu
Nhận diện biển số xe là một công nghệ quan trọng trong các hệ thống giám sát, quản lý giao thông, kiểm tra an ninh, thu phí tự động, và quản lý bãi đỗ xe tự động.

Công nghệ này sử dụng Object Detection (Phát hiện đối tượng) kết hợp với OCR-Optical Character Recognition (nhận dạng ký tự quang học) để tự động phát hiện và đọc biển số từ hình ảnh hoặc video.

## Mục tiêu đề tài
* Phát hiện biển số xe trong ảnh
* Nhận diện ký tự trên biển số 
* Triển khai API demo cho hệ thống nhận diện biển số xe

## Thành viên nhóm 
* A46350 Trần Huyền Trang
* A46483 Nguyễn Thị Thùy Trang
* A45510 Phạm Thanh Mai

## Công nghệ sử dụng
* Ngôn ngữ lập trình: Python
* Thư viện chính: 
* Mô hình: YOLO11, CRNN (CNN + RNN + CTCLoss)

## Bộ dữ liệu sử dụng
Chúng tôi sử dụng bộ dữ liệu thô từ công ty GreenParking và thực hiện gán nhãn bằng Roboflow. Bạn có thể xem bộ dữ liệu sau khi được gán nhãn [tại đây](https://universe.roboflow.com/cm-h8pey/biensoxe-mtw44).

## Hướng dẫn chạy demo
...
## Kết quả và Đánh giá
Mô hình được đánh giá dựa trên các chỉ số:
* Accuracy
* Character-level Accuracy
* Recall
* Precision
* F1-score
* Inference Time
* IoU (Intersection over Union) 
