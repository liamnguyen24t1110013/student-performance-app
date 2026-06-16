import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="Ứng dụng ML trong y tế",
    page_icon="❤️",
    layout="wide"
)

def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

@st.cache_data
def load_data():
    return pd.read_csv("heart_health_data.csv")

df = load_data()

feature_cols = [
    "tuoi",
    "huyet_ap",
    "cholesterol",
    "nhip_tim_toi_da",
    "duong_huyet_cao",
    "so_gio_tap_the_duc_tuan",
    "dau_nguc"
]

X = df[feature_cols]
y = df["nguy_co_benh_tim"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Hồi quy Logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500))
    ]),
    "Cây quyết định": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=180, random_state=42, max_depth=6)
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    results.append({
        "Thuật toán": name,
        "Accuracy (%)": round(accuracy_score(y_test, pred) * 100, 2),
        "F1-score": round(f1_score(y_test, pred, pos_label="Nguy cơ cao"), 3)
    })

result_df = pd.DataFrame(results)
best_name = result_df.sort_values("Accuracy (%)", ascending=False).iloc[0]["Thuật toán"]
best_model = models[best_name]
best_pred = predictions[best_name]
accuracy = accuracy_score(y_test, best_pred)
f1 = f1_score(y_test, best_pred, pos_label="Nguy cơ cao")
high_rate = (df["nguy_co_benh_tim"] == "Nguy cơ cao").mean() * 100

with st.sidebar:
    st.markdown("## ❤️ Health ML")
    st.caption("Dự đoán nguy cơ bệnh tim")
    st.markdown("---")
    page = st.radio(
        "Chọn mục",
        [
            "🏠 Tổng quan",
            "🧠 3 thuật toán",
            "🩺 Nhập dữ liệu sức khỏe",
            "📊 Phân tích dữ liệu",
            "🤖 So sánh mô hình",
            "📋 Dữ liệu CSV"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.warning("Kết quả chỉ phục vụ học tập, không thay thế tư vấn y tế.")

def hero():
    st.markdown("""
    <div class="hero">
        <h1>Ứng dụng Machine Learning trong y tế: Dự đoán nguy cơ bệnh tim ❤️</h1>
        <p>
        App sử dụng 3 thuật toán học máy gồm Hồi quy Logistic, Cây quyết định và Random Forest
        để phân tích dữ liệu sức khỏe, so sánh mô hình và dự đoán nguy cơ bệnh tim.
        </p>
        <span class="badge">Hồi quy Logistic</span>
        <span class="badge">Cây quyết định</span>
        <span class="badge">Random Forest</span>
        <span class="badge">Python</span>
        <span class="badge">Streamlit</span>
    </div>
    """, unsafe_allow_html=True)

def show_metrics():
    st.markdown('<div class="section-title">Tổng quan hệ thống</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (len(df), "Dòng dữ liệu", "🗃️"),
        (len(feature_cols), "Biến sức khỏe", "🩺"),
        (f"{accuracy*100:.1f}%", f"Accuracy tốt nhất ({best_name})", "✅"),
        (f"{high_rate:.1f}%", "Tỷ lệ nguy cơ cao", "❤️")
    ]
    for col, (num, label, icon) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size:32px">{icon}</div>
                <div class="metric-number">{num}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def page_overview():
    hero()
    show_metrics()
    st.markdown('<div class="section-title">Giới thiệu đề tài</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h3>🎯 Mục tiêu nghiên cứu</h3>
        <p>
        Đề tài tìm hiểu ứng dụng của học máy trong lĩnh vực y tế, cụ thể là bài toán
        dự đoán nguy cơ bệnh tim dựa trên các chỉ số sức khỏe như tuổi, huyết áp,
        cholesterol, nhịp tim, đường huyết, thói quen vận động và triệu chứng đau ngực.
        </p>
        <p>
        Ứng dụng giúp minh họa quy trình khoa học dữ liệu: thu thập dữ liệu, tiền xử lý,
        huấn luyện mô hình, đánh giá kết quả và triển khai giao diện Web bằng Streamlit.
        </p>
    </div>
    """, unsafe_allow_html=True)

def page_algorithms():
    st.markdown('<div class="section-title">3 thuật toán được sử dụng</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    algo_cards = [
        ("📈", "Hồi quy Logistic", "Dùng cho bài toán phân loại, phù hợp dự đoán nguy cơ thấp/cao dựa trên xác suất."),
        ("🌳", "Cây quyết định", "Mô hình dạng cây, dễ hiểu và dễ giải thích các điều kiện ra quyết định."),
        ("🌲", "Random Forest", "Kết hợp nhiều cây quyết định để tăng độ ổn định và độ chính xác của mô hình.")
    ]
    for col, (icon, title, desc) in zip([a1, a2, a3], algo_cards):
        with col:
            st.markdown(f"""
            <div class="algo-box">
                <div style="font-size:44px">{icon}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Bảng so sánh lý thuyết</div>', unsafe_allow_html=True)
    compare_theory = pd.DataFrame({
        "Thuật toán": ["Hồi quy Logistic", "Cây quyết định", "Random Forest"],
        "Ưu điểm": [
            "Đơn giản, nhanh, dễ dùng cho phân loại nhị phân",
            "Dễ hiểu, dễ trực quan hóa điều kiện quyết định",
            "Độ chính xác tốt, giảm overfitting so với một cây đơn"
        ],
        "Hạn chế": [
            "Khó mô hình hóa quan hệ phi tuyến phức tạp",
            "Dễ overfitting nếu cây quá sâu",
            "Khó giải thích hơn cây quyết định đơn"
        ],
        "Ứng dụng": [
            "Dự đoán nguy cơ bệnh, phân loại spam",
            "Hỗ trợ ra quyết định, phân loại bệnh nhân",
            "Y tế, tài chính, giáo dục, thương mại điện tử"
        ]
    })
    st.dataframe(compare_theory, use_container_width=True)

def page_predict():
    st.markdown('<div class="section-title">Nhập dữ liệu sức khỏe để dự đoán</div>', unsafe_allow_html=True)
    form_col, result_col = st.columns([1, 1])

    with form_col:
        st.markdown("### 📝 Thông tin sức khỏe")
        selected_model_name = st.selectbox("Chọn thuật toán dự đoán", list(models.keys()), index=2)
        tuoi = st.number_input("Tuổi", min_value=20, max_value=90, value=45, step=1)
        huyet_ap = st.number_input("Huyết áp", min_value=80, max_value=200, value=130, step=1)
        cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=220, step=1)
        nhip_tim = st.number_input("Nhịp tim tối đa", min_value=60, max_value=220, value=145, step=1)
        duong_huyet = st.selectbox("Đường huyết cao", ["Không", "Có"])
        tap_the_duc = st.number_input("Số giờ tập thể dục mỗi tuần", min_value=0, max_value=10, value=3, step=1)
        dau_nguc = st.selectbox("Có đau ngực không?", ["Không", "Có"])
        predict_btn = st.button("🚀 Dự đoán nguy cơ")

    with result_col:
        st.markdown("### 🎯 Kết quả dự đoán")
        input_df = pd.DataFrame([{
            "tuoi": tuoi,
            "huyet_ap": huyet_ap,
            "cholesterol": cholesterol,
            "nhip_tim_toi_da": nhip_tim,
            "duong_huyet_cao": 1 if duong_huyet == "Có" else 0,
            "so_gio_tap_the_duc_tuan": tap_the_duc,
            "dau_nguc": 1 if dau_nguc == "Có" else 0
        }])

        if predict_btn:
            selected_model = models[selected_model_name]
            result = selected_model.predict(input_df)[0]
            proba = selected_model.predict_proba(input_df).max() * 100

            if result == "Nguy cơ thấp":
                st.markdown(f"""
                <div class="result-low">
                    <h1>NGUY CƠ THẤP ✅</h1>
                    <p>Thuật toán: <b>{selected_model_name}</b></p>
                    <p>Độ tin cậy dự đoán: <b>{proba:.1f}%</b></p>
                    <p>Duy trì lối sống lành mạnh và theo dõi sức khỏe định kỳ.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-high">
                    <h1>NGUY CƠ CAO ⚠️</h1>
                    <p>Thuật toán: <b>{selected_model_name}</b></p>
                    <p>Độ tin cậy dự đoán: <b>{proba:.1f}%</b></p>
                    <p>Nên kiểm tra sức khỏe và tham khảo ý kiến bác sĩ.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="notice">
                <b>Hướng dẫn:</b><br>
                Chọn thuật toán, nhập các chỉ số sức khỏe rồi bấm nút dự đoán.
            </div>
            """, unsafe_allow_html=True)

def page_analysis():
    st.markdown('<div class="section-title">Phân tích dữ liệu y tế</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📊 Phân bố nguy cơ bệnh tim")
        counts = df["nguy_co_benh_tim"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        st.pyplot(fig)

    with c2:
        st.markdown("### 📈 Chỉ số trung bình theo nhóm nguy cơ")
        mean_df = df.groupby("nguy_co_benh_tim")[["huyet_ap", "cholesterol", "nhip_tim_toi_da"]].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        mean_df.plot(kind="bar", ax=ax)
        ax.set_ylabel("Giá trị trung bình")
        ax.set_xlabel("Nhóm nguy cơ")
        ax.grid(axis="y", alpha=0.2)
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### 🧓 Tuổi và cholesterol")
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = np.where(df["nguy_co_benh_tim"] == "Nguy cơ cao", 1, 0)
        ax.scatter(df["tuoi"], df["cholesterol"], c=colors, alpha=0.75)
        ax.set_xlabel("Tuổi")
        ax.set_ylabel("Cholesterol")
        ax.grid(alpha=0.2)
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        st.pyplot(fig)

    with c4:
        st.markdown("### 🔥 Tương quan giữa các biến")
        corr = df[feature_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 4.8))
        im = ax.imshow(corr)
        labels = ["Tuổi", "Huyết áp", "Cholesterol", "Nhịp tim", "Đường huyết", "Tập TD", "Đau ngực"]
        ax.set_xticks(range(len(feature_cols)))
        ax.set_yticks(range(len(feature_cols)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax)
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        st.pyplot(fig)

def page_model_compare():
    st.markdown('<div class="section-title">So sánh kết quả 3 thuật toán</div>', unsafe_allow_html=True)
    compare_col, cm_col = st.columns([1.15, 1])

    with compare_col:
        st.markdown("### 📊 Bảng đánh giá mô hình")
        st.dataframe(result_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(result_df["Thuật toán"], result_df["Accuracy (%)"])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=0.2)
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        st.pyplot(fig)

    with cm_col:
        st.markdown(f"### 🔍 Ma trận nhầm lẫn: {best_name}")
        labels = ["Nguy cơ thấp", "Nguy cơ cao"]
        cm = confusion_matrix(y_test, best_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.imshow(cm)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=16, fontweight="bold")
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        st.pyplot(fig)
        st.markdown(f"**Accuracy:** {accuracy*100:.2f}%")
        st.markdown(f"**F1-score:** {f1:.3f}")

    st.markdown("### ⭐ Mức độ quan trọng của biến theo Random Forest")
    rf_model = models["Random Forest"]
    imp = pd.DataFrame({
        "Biến": ["Tuổi", "Huyết áp", "Cholesterol", "Nhịp tim", "Đường huyết", "Tập thể dục", "Đau ngực"],
        "Độ quan trọng": rf_model.feature_importances_
    }).sort_values("Độ quan trọng", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp["Biến"], imp["Độ quan trọng"])
    ax.set_xlabel("Feature importance")
    ax.grid(axis="x", alpha=0.2)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    st.pyplot(fig)

def page_data():
    st.markdown('<div class="section-title">Dữ liệu thực nghiệm</div>', unsafe_allow_html=True)
    st.write("Bảng dữ liệu mẫu được lưu trong file `heart_health_data.csv` và dùng để huấn luyện 3 mô hình.")
    st.dataframe(df, use_container_width=True)

if page == "🏠 Tổng quan":
    page_overview()
elif page == "🧠 3 thuật toán":
    page_algorithms()
elif page == "🩺 Nhập dữ liệu sức khỏe":
    page_predict()
elif page == "📊 Phân tích dữ liệu":
    page_analysis()
elif page == "🤖 So sánh mô hình":
    page_model_compare()
elif page == "📋 Dữ liệu CSV":
    page_data()

st.markdown("""
<br>
<div style="text-align:center;color:#cbd5e1;padding:20px;">
    Made with ❤️ using Python, Streamlit & Scikit-learn<br>
    Đề tài: Ứng dụng Machine Learning trong y tế - Dự đoán nguy cơ bệnh tim
</div>
""", unsafe_allow_html=True)
