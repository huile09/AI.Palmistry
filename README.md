## Hệ Thống Nhận Diện Chỉ Tay (Palmistry Recognition)

Dự án này sử dụng mạng nơ-ron tích chập (CNN) được xây dựng để phân loại và nhận diện các đường nét đặc trưng trên lòng bàn tay (chỉ tay). Mô hình được tối ưu hóa để nhận diện chính xác các cấu trúc đường chỉ tay chính phục vụ cho bài toán nhân trắc học hoặc phân tích tự động.

## Tải Xuống Mô Hình Pre-trained và Dữ liệu Test

Do giới hạn dung lượng file trên GitHub, file trọng số mô hình .keras được lưu trữ tại Google Drive. Bạn có thể tải về để sử dụng ngay mà không cần huấn luyện lại:

👉 [**Tải file mo_hinh_nhan_dien_tien_50_epochs2.h5 tại đây**](https://drive.google.com/file/d/1ZDzOwus3XsSFWFRsl3ITRnD4vD6mgpTI/view?usp=drive_link)

👉 [**Tải file ảnh test model tại đây**](https://drive.google.com/file/d/1v3YgENMoyxB56whZoqcXzt-RL__WG3x-/view?usp=drive_link)

## 🛠️ Cách Chạy Mô Hình
Bước 1. Tải file ảnh test model về rồi đưa lên Google Drive
Bước 2. Thực hiện giải nén
```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -o -q "/content/drive/MyDrive/palmistry.zip" -d "/content/dataset"
```
Bước 3. Nhập model và test
```python
from tensorflow.keras.models import load_model

# Load model từ file đã tải về 
model = load_model('best_palmistry_model.pt')
```
Bước 4. Test ảnh
```python
drive_model_path = '/content/drive/MyDrive/best_palmistry_model.pt'
if os.path.exists(drive_model_path):
    model = YOLO(drive_model_path)

    vietnamese_names = {0: 'Sự nghiệp', 1: 'Sinh đạo', 2: 'Trí đạo', 3: 'Tam đạo'}
    model.model.names = vietnamese_names

    test_image_dir = '/content/dataset/test/images'
    test_images = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    sample_images = random.sample(test_images, min(3, len(test_images)))

    for img_path in sample_images:
        results = model.predict(source=img_path, conf=0.25, imgsz=640, verbose=False)
        annotated_img = results[0].plot()

        print(f"\nĐường chỉ tay nhận diện trên ảnh: {os.path.basename(img_path)}")
        cv2_imshow(annotated_img)
else:
    print("Không tìm thấy file tại đường dẫn")
```
