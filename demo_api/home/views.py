from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import base64
import cv2
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from . import plate_detec
from . import detection
from . import LicensePlateRecognizer

model = detection.load_model('D:/mine/HocTap/ThiGiacMay/git/demo_api/home/best.pt')

# Create your views here.
def get_home(request):
    return render(request, 'lisence_plate.html')

def upload_image(request):
    if request.method == 'POST' and 'imageInput' in request.FILES:
        # Nhận ảnh từ form
        image_file = request.FILES['imageInput']
        
        # Lưu ảnh vào thư mục tạm thời
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        file_url = fs.url(filename)

        # Lấy đường dẫn thực tế của ảnh đã tải lên
        image_path = fs.path(filename)
        print(f"File được tải lên: {image_path}")

        # Xử lý ảnh và nhận diện biển số
        result_img, result_text = plate_detec.predict(image_path)  # Lấy ảnh với bounding box và văn bản nhận diện

        # Chuyển ảnh nhận diện (result_img) sang base64
        _, buffer = cv2.imencode('.jpg', result_img)
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        # Trả kết quả về cho client dưới dạng JsonResponse
        return JsonResponse({
            'result': result_text, 
            'image_url': file_url, 
            'bounding_box_image': 'data:image/jpeg;base64,' + result_base64  # Trả về ảnh đã nhận diện dưới dạng base64
        })
        
    return render(request, 'lisence_plate.html')

@csrf_exempt
def process_video(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        image_data = data.split('data:image/jpeg;base64,')[1]  # Cắt ảnh từ base64

        # Giải mã base64 thành ảnh
        image = base64.b64decode(image_data)
        np_img = np.frombuffer(image, dtype=np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Dự đoán và nhận diện biển số xe từ ảnh
        cropped_plates, result_img = detection.detect_plate(model, img)  # Hàm detect_plate chạy YOLO

        # Chuyển kết quả trả về thành base64
        _, buffer = cv2.imencode('.jpg', result_img)
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        return JsonResponse({
            'result': 'Detected Plate',
            'image_url': 'data:image/jpeg;base64,' + result_base64  # Trả về ảnh đã nhận diện
        })

    return JsonResponse({'error': 'Invalid request'}, status=400)