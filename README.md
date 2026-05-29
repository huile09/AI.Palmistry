## Hệ Thống Nhận Diện Chỉ Tay (Palmistry Recognition)

Dự án này sử dụng mạng nơ-ron tích chập (CNN) được xây dựng để phân loại và nhận diện các đường nét đặc trưng trên lòng bàn tay (chỉ tay). Mô hình được tối ưu hóa để nhận diện chính xác các cấu trúc đường chỉ tay chính phục vụ cho bài toán nhân trắc học hoặc phân tích tự động.

## Tải Xuống Mô Hình Pre-trained

Do giới hạn dung lượng file trên GitHub, file trọng số mô hình .keras được lưu trữ tại Google Drive. Bạn có thể tải về để sử dụng ngay mà không cần huấn luyện lại:

👉 [**Tải file mo_hinh_nhan_dien_tien_50_epochs2.h5 tại đây**](https://drive.google.com/file/d/1ZDzOwus3XsSFWFRsl3ITRnD4vD6mgpTI/view?usp=drive_link)
## 📂 Cấu Trúc Thư Mục Dự Án

- AI_Palmistry.ipynb : File Google Colab chứa toàn bộ mã nguồn từ xử lý dữ liệu, chuẩn hóa hình ảnh lòng bàn tay đến huấn luyện.
- test_file/ : Thư mục chứa các hình ảnh chỉ tay thực tế dùng để kiểm thử (Test).
- README.md : Hướng dẫn và thông tin dự án.

## 🛠️ Cách Chạy Mô Hình

Bạn có thể nạp lại mô hình trong môi trường Python/Colab bằng lệnh sau:

```python
from tensorflow.keras.models import load_model

# Load model từ file đã tải về 
model = load_model('best_palmistry_model.pt')

# Dự đoán ảnh mới
# predictions = model.predict(img_prepared)
print("✅ Mô hình Palmistry đã sẵn sàng!")
