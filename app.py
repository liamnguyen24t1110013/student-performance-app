import streamlit as st

st.set_page_config(page_title="Machine Learning Demo", layout="wide")

st.title("TÌM HIỂU MỘT SỐ THUẬT TOÁN HỌC MÁY")
st.write("Ứng dụng minh họa được xây dựng bằng Python và Streamlit.")

st.header("Các thuật toán phổ biến")
st.markdown("""
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
""")

st.header("Ứng dụng trong cuộc sống")
st.markdown("""
- Dự đoán kết quả học tập
- Nhận diện khuôn mặt
- Gợi ý sản phẩm
- Phân loại email spam
- Dự đoán giá nhà
""")

st.success("Ứng dụng Streamlit đang hoạt động thành công!")
