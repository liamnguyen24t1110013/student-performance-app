import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Một số thuật toán học máy",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #1e3a8a;
    font-size: 42px;
    font-weight: 800;
}
.sub-title {
    text-align: center;
    color: #334155;
    font-size: 20px;
}
.card {
    background-color: #f8fafc;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    margin-bottom: 12px;
}
.metric-box {
    background-color: #eef2ff;
    padding: 14px;
    border-radius: 14px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Tìm hiểu một số thuật toán học máy</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng minh họa bằng Python và Streamlit</div>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Cấu hình mô hình")

dataset_name = st.sidebar.selectbox(
    "Chọn bộ dữ liệu thực tế",
    ["Iris - Phân loại loài hoa", "Wine - Phân loại rượu vang", "Breast Cancer - Phân loại khối u"]
)

algorithm_name = st.sidebar.selectbox(
    "Chọn thuật toán học máy",
    ["KNN", "Decision Tree", "Random Forest", "Logistic Regression"]
)

test_size = st.sidebar.slider("Tỷ lệ dữ liệu kiểm tra", 0.1, 0.5, 0.2, 0.05)
random_state = 42

def load_dataset(name):
    if name.startswith("Iris"):
        data = load_iris()
        description = "Bộ dữ liệu Iris dùng để phân loại 3 loài hoa dựa trên chiều dài, chiều rộng đài hoa và cánh hoa."
    elif name.startswith("Wine"):
        data = load_wine()
        description = "Bộ dữ liệu Wine dùng để phân loại rượu vang dựa trên các chỉ số hóa học."
    else:
        data = load_breast_cancer()
        description = "Bộ dữ liệu Breast Cancer dùng để phân loại khối u lành tính hoặc ác tính dựa trên các đặc trưng y khoa."
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["target_name"] = df["target"].apply(lambda x: data.target_names[x])
    return data, df, description

data, df, dataset_description = load_dataset(dataset_name)

X = df[data.feature_names]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

def build_model(name):
    if name == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ])
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=5, random_state=random_state)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=120, random_state=random_state)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=random_state))
    ])

model = build_model(algorithm_name)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Giới thiệu",
    "📊 Dữ liệu",
    "📈 Trực quan hóa",
    "🤖 Huấn luyện mô hình",
    "🧪 Dự đoán thử"
])

with tab1:
    st.markdown("## 1. Giới thiệu đề tài")
    st.markdown("""
    Đề tài **“Tìm hiểu về một số thuật toán học máy và ứng dụng trong cuộc sống”** 
    tập trung trình bày các thuật toán học máy cơ bản, sau đó triển khai một ứng dụng minh họa 
    bằng Python và Streamlit.

    Ứng dụng cho phép người dùng:
    - Chọn bộ dữ liệu thực tế.
    - Chọn thuật toán học máy.
    - Huấn luyện mô hình.
    - Xem biểu đồ trực quan hóa dữ liệu.
    - Đánh giá mô hình bằng Accuracy, Precision, Recall và F1-score.
    - Nhập dữ liệu mới để mô hình dự đoán.
    """)

    st.markdown("## 2. Một số thuật toán được sử dụng")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
        <b>KNN</b><br>
        Phân loại đối tượng mới dựa vào các điểm dữ liệu gần nhất.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <b>Decision Tree</b><br>
        Mô hình cây quyết định, chia dữ liệu theo các điều kiện để đưa ra dự đoán.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <b>Random Forest</b><br>
        Kết hợp nhiều cây quyết định để tăng độ chính xác và giảm quá khớp.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <b>Logistic Regression</b><br>
        Thuật toán phân loại phổ biến, thường dùng trong các bài toán nhị phân hoặc đa lớp.
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("## Thông tin bộ dữ liệu")
    st.info(dataset_description)

    col1, col2, col3 = st.columns(3)
    col1.metric("Số dòng dữ liệu", df.shape[0])
    col2.metric("Số thuộc tính đầu vào", len(data.feature_names))
    col3.metric("Số lớp cần phân loại", len(data.target_names))

    st.markdown("### 5 dòng dữ liệu đầu tiên")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### Thống kê mô tả")
    st.dataframe(df[data.feature_names].describe(), use_container_width=True)

with tab3:
    st.markdown("## Trực quan hóa dữ liệu")

    col1, col2 = st.columns(2)

    with col1:
        count_df = df["target_name"].value_counts().reset_index()
        count_df.columns = ["Lớp", "Số lượng"]
        fig1 = px.bar(count_df, x="Lớp", y="Số lượng", title="Biểu đồ 1: Số lượng mẫu theo từng lớp")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.pie(count_df, names="Lớp", values="Số lượng", title="Biểu đồ 2: Tỷ lệ các lớp trong dữ liệu")
        st.plotly_chart(fig2, use_container_width=True)

    first_feature = data.feature_names[0]
    second_feature = data.feature_names[1]

    fig3 = px.histogram(df, x=first_feature, color="target_name", title=f"Biểu đồ 3: Phân bố {first_feature}")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.scatter(
        df, x=first_feature, y=second_feature, color="target_name",
        title=f"Biểu đồ 4: Mối quan hệ giữa {first_feature} và {second_feature}"
    )
    st.plotly_chart(fig4, use_container_width=True)

    corr = df[data.feature_names].corr()
    fig5 = px.imshow(corr, text_auto=True, title="Biểu đồ 5: Ma trận tương quan giữa các thuộc tính")
    st.plotly_chart(fig5, use_container_width=True)

with tab4:
    st.markdown("## Huấn luyện và đánh giá mô hình")
    st.write(f"**Bộ dữ liệu:** {dataset_name}")
    st.write(f"**Thuật toán:** {algorithm_name}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1-score", f"{f1:.2%}")

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=data.target_names, columns=data.target_names)
    fig_cm = px.imshow(cm_df, text_auto=True, title="Ma trận nhầm lẫn")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### Báo cáo phân loại")
    report = classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

with tab5:
    st.markdown("## Nhập dữ liệu mới để dự đoán")

    input_values = []
    cols = st.columns(2)
    for i, feature in enumerate(data.feature_names):
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        with cols[i % 2]:
            value = st.number_input(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
            input_values.append(value)

    if st.button("Dự đoán kết quả"):
        input_df = pd.DataFrame([input_values], columns=data.feature_names)
        prediction = model.predict(input_df)[0]
        proba_text = ""

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            proba_text = f"Độ tin cậy cao nhất: {np.max(probs):.2%}"

        st.success(f"Kết quả dự đoán: {data.target_names[prediction]}")
        if proba_text:
            st.info(proba_text)

st.markdown("---")
st.caption("Ứng dụng minh họa cho tiểu luận học phần Nhập môn Khoa học dữ liệu.")
