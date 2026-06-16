# Ứng dụng Machine Learning trong y tế: Dự đoán nguy cơ bệnh tim

Ứng dụng Streamlit cho tiểu luận học phần Nhập môn Khoa học dữ liệu.

## Thuật toán sử dụng

- Hồi quy Logistic
- Cây quyết định
- Random Forest

## File trong dự án

- `app.py`: mã nguồn ứng dụng Streamlit
- `style.css`: giao diện CSS
- `heart_health_data.csv`: dữ liệu sức khỏe
- `requirements.txt`: thư viện cần cài
- `README.md`: hướng dẫn

## Chức năng

- Nhập dữ liệu sức khỏe trực tiếp
- Chọn thuật toán để dự đoán
- So sánh Accuracy và F1-score của 3 thuật toán
- Trực quan hóa dữ liệu bằng biểu đồ
- Đánh giá mô hình bằng Confusion Matrix
- Hiển thị bảng dữ liệu CSV

## Lưu ý

Ứng dụng chỉ phục vụ mục đích học tập, không thay thế chẩn đoán hoặc tư vấn y tế chuyên môn.

## Chạy ứng dụng

```bash
streamlit run app.py
```
