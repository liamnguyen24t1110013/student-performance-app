import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Thuật toán học máy & ứng dụng",
    page_icon="🤖",
    layout="wide"
)

# ===================== CSS =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 12% 15%, rgba(124, 58, 237, 0.35), transparent 28%),
        radial-gradient(circle at 80% 5%, rgba(14, 165, 233, 0.30), transparent 28%),
        radial-gradient(circle at 75% 80%, rgba(34, 197, 94, 0.16), transparent 28%),
        linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.92);
    border-right: 1px solid rgba(148, 163, 184, 0.20);
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

h1, h2, h3, h4, h5, h6, p, li, span, div {
    color: #f8fafc;
}

.hero {
    padding: 38px;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(37, 99, 235, 0.55), rgba(147, 51, 234, 0.42)),
        url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 25px 70px rgba(0,0,0,0.45);
    min-height: 310px;
}

.hero h1 {
    font-size: 46px;
    line-height: 1.15;
    margin-bottom: 18px;
    color: #ffffff !important;
    text-shadow: 0 3px 15px rgba(0,0,0,0.45);
}

.hero p {
    font-size: 18px;
    max-width: 850px;
    color: #e0f2fe !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.42);
}

.glass-card {
    padding: 24px;
    border-radius: 22px;
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(255,255,255,0.13);
    box-shadow: 0 16px 45px rgba(0,0,0,0.35);
    backdrop-filter: blur(14px);
    min-height: 150px;
}

.algo-card {
    padding: 22px;
    border-radius: 22px;
    background: linear-gradient(145deg, rgba(30,41,59,0.92), rgba(15,23,42,0.86));
    border: 1px solid rgba(148,163,184,0.22);
    min-height: 240px;
    box-shadow: 0 15px 38px rgba(0,0,0,0.25);
}

.algo-card h3 {
    color: #ffffff !important;
    font-size: 21px;
}

.algo-card p {
    color: #cbd5e1 !important;
    font-size: 14px;
}

.metric-box {
    padding: 20px;
    border-radius: 20px;
    background: rgba(15,23,42,0.82);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 14px 35px rgba(0,0,0,0.25);
}

.metric-number {
    font-size: 34px;
    font-weight: 800;
    color: #ffffff !important;
}

.metric-label {
    font-size: 14px;
    color: #cbd5e1 !important;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #ffffff !important;
    margin-top: 30px;
    margin-bottom: 14px;
}

.badge {
    display: inline-block;
    padding: 7px 12px;
    margin: 5px 5px 5px 0;
    border-radius: 999px;
    background: rgba(59, 130, 246, 0.25);
    border: 1px solid rgba(125, 211, 252, 0.35);
    color: #e0f2fe !important;
    font-size: 13px;
    font-weight: 700;
}

.result-good {
    padding: 25px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(34,197,94,0.25), rgba(14,165,233,0.16));
    border: 1px solid rgba(74,222,128,0.45);
    text-align: center;
}

.info-box {
    padding: 22px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(245,158,11,0.20), rgba(59,130,246,0.12));
    border: 1px solid rgba(251,191,36,0.28);
}

div.stButton > button {
    width: 100%;
    border-radius: 15px;
    padding: 14px 18px;
    font-weight: 800;
    color: #ffffff;
    background: linear-gradient(90deg, #2563eb, #9333ea);
    border: 0;
}

div.stButton > button:hover {
    color: #ffffff;
    border: 0;
    filter: brightness(1.15);
}

[data-testid="stMetricValue"] {
    color: #ffffff;
}

.stDataFrame {
    border-radius: 14px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ===================== DATA =====================
algorithms = pd.DataFrame({
    "Thuật toán": ["Linear Regression", "Logistic Regression", "Decision Tree", "KNN", "Random Forest"],
    "Loại bài toán": ["Hồi quy", "Phân loại", "Phân loại/Hồi quy", "Phân loại/Hồi quy", "Phân loại/Hồi quy"],
    "Dễ hiểu": [5, 4, 5, 4, 3],
    "Tốc độ": [5, 5, 4, 3, 3],
    "Độ chính xác": [3, 4, 4, 4, 5],
    "Dễ giải thích": [5, 4, 5, 3, 3],
    "Ứng dụng": [
        "Dự đoán giá nhà, doanh thu, nhiệt độ",
        "Phân loại email spam, dự đoán đậu/rớt",
        "Phân loại khách hàng, hỗ trợ ra quyết định",
        "Nhận dạng mẫu, hệ thống gợi ý",
        "Y tế, tài chính, giáo dục, thương mại điện tử"
    ]
})

life_apps = pd.DataFrame({
    "Lĩnh vực": ["Y tế", "Tài chính", "Giáo dục", "Thương mại điện tử", "Giao thông", "An ninh"],
    "Mức độ ứng dụng": [92, 88, 84, 95, 78, 82],
    "Ví dụ": [
        "Hỗ trợ chẩn đoán bệnh",
        "Phát hiện gian lận giao dịch",
        "Dự đoán kết quả học tập",
        "Gợi ý sản phẩm",
        "Dự đoán ùn tắc giao thông",
        "Nhận diện khuôn mặt"
    ]
})

# Iris machine learning demo
iris = load_iris()
X = pd.DataFrame(
    iris.data,
    columns=["Chiều dài đài hoa", "Chiều rộng đài hoa", "Chiều dài cánh hoa", "Chiều rộng cánh hoa"]
)
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=300))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)
}

scores = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores[name] = accuracy_score(y_test, y_pred)
    predictions[name] = y_pred

best_name = max(scores, key=scores.get)
best_model = models[best_name]

# Linear regression illustration
np.random.seed(7)
x_lr = np.arange(1, 22)
y_lr = 2.9 * x_lr + np.random.normal(0, 4, len(x_lr)) + 12
linear_model = LinearRegression()
linear_model.fit(x_lr.reshape(-1, 1), y_lr)
y_line = linear_model.predict(x_lr.reshape(-1, 1))


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("## 🤖 ML Dashboard")
    st.caption("Đề tài tiểu luận")
    st.markdown("---")
    st.markdown("🏠 **Tổng quan**")
    st.markdown("🧠 **5 thuật toán học máy**")
    st.markdown("📊 **Biểu đồ so sánh**")
    st.markdown("🌍 **Ứng dụng đời sống**")
    st.markdown("🎯 **Demo mô hình Python**")
    st.markdown("---")
    st.success("App đã sẵn sàng để đưa vào tiểu luận.")


# ===================== HERO =====================
st.markdown("""
<div class="hero">
    <h1>Tìm hiểu về một số thuật toán học máy và ứng dụng trong cuộc sống 🚀</h1>
    <p>
    Ứng dụng Streamlit minh họa Machine Learning bằng giao diện trực quan:
    tổng quan lý thuyết, so sánh thuật toán, biểu đồ phân tích và demo mô hình Python.
    </p>
    <span class="badge">Linear Regression</span>
    <span class="badge">Logistic Regression</span>
    <span class="badge">Decision Tree</span>
    <span class="badge">KNN</span>
    <span class="badge">Random Forest</span>
</div>
""", unsafe_allow_html=True)


# ===================== METRICS =====================
st.markdown('<div class="section-title">Tổng quan ứng dụng</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
metric_data = [
    ("5", "Thuật toán học máy", "🧠"),
    ("6", "Lĩnh vực ứng dụng", "🌍"),
    ("6+", "Biểu đồ trực quan", "📊"),
    (f"{scores[best_name]*100:.1f}%", "Accuracy demo tốt nhất", "✅")
]

for col, (num, label, icon) in zip([m1, m2, m3, m4], metric_data):
    with col:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:32px">{icon}</div>
            <div class="metric-number">{num}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ===================== ALGORITHM CARDS =====================
st.markdown('<div class="section-title">1. Các thuật toán học máy tiêu biểu</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
algo_cards = [
    ("📈", "Linear Regression", "Dự đoán giá trị liên tục như giá nhà, doanh thu hoặc nhiệt độ.", "Hồi quy"),
    ("📧", "Logistic Regression", "Phân loại dữ liệu, ví dụ email spam/không spam hoặc đậu/rớt.", "Phân loại"),
    ("🌳", "Decision Tree", "Mô hình dạng cây, dễ hiểu, dễ giải thích và phù hợp ra quyết định.", "Cây quyết định"),
    ("📍", "KNN", "Dự đoán dựa trên các điểm dữ liệu gần nhất trong không gian đặc trưng.", "Khoảng cách"),
    ("🌲", "Random Forest", "Kết hợp nhiều cây quyết định để tăng độ chính xác và giảm overfitting.", "Tập hợp")
]

for col, (icon, title, desc, tag) in zip([col1, col2, col3, col4, col5], algo_cards):
    with col:
        st.markdown(f"""
        <div class="algo-card">
            <div style="font-size:42px">{icon}</div>
            <h3>{title}</h3>
            <p>{desc}</p>
            <span class="badge">{tag}</span>
        </div>
        """, unsafe_allow_html=True)


# ===================== CHARTS =====================
st.markdown('<div class="section-title">2. So sánh và trực quan hóa thuật toán</div>', unsafe_allow_html=True)

chart1, chart2 = st.columns([1.25, 1])

with chart1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 So sánh thuật toán theo tiêu chí")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(algorithms))
    width = 0.22
    ax.bar(x - width, algorithms["Tốc độ"], width, label="Tốc độ")
    ax.bar(x, algorithms["Độ chính xác"], width, label="Độ chính xác")
    ax.bar(x + width, algorithms["Dễ giải thích"], width, label="Dễ giải thích")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms["Thuật toán"], rotation=22, ha="right")
    ax.set_ylim(0, 5.6)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with chart2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🧩 Nhóm bài toán")
    task_counts = algorithms["Loại bài toán"].value_counts()
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.pie(task_counts.values, labels=task_counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

chart3, chart4 = st.columns([1, 1])

with chart3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🌍 Mức độ ứng dụng trong đời sống")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(life_apps["Lĩnh vực"], life_apps["Mức độ ứng dụng"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mức độ ứng dụng (%)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with chart4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Minh họa Linear Regression")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.scatter(x_lr, y_lr, label="Dữ liệu")
    ax.plot(x_lr, y_line, label="Đường hồi quy")
    ax.set_xlabel("Biến đầu vào giả lập")
    ax.set_ylabel("Giá trị dự đoán")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# ===================== LIFE APPLICATIONS =====================
st.markdown('<div class="section-title">3. Ứng dụng học máy trong cuộc sống</div>', unsafe_allow_html=True)

icons = ["🏥", "🏦", "🎓", "🛒", "🚗", "🔐"]
app_cols = st.columns(3)

for i, row in life_apps.iterrows():
    with app_cols[i % 3]:
        st.markdown(f"""
        <div class="glass-card">
            <h2>{icons[i]} {row["Lĩnh vực"]}</h2>
            <p style="color:#cbd5e1 !important">{row["Ví dụ"]}</p>
            <p><b>Mức độ ứng dụng:</b> {row["Mức độ ứng dụng"]}%</p>
        </div>
        <br>
        """, unsafe_allow_html=True)


# ===================== ML DEMO =====================
st.markdown('<div class="section-title">4. Demo mô hình học máy bằng Python</div>', unsafe_allow_html=True)

demo_left, demo_right = st.columns([1.35, 1])

with demo_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🌸 Bài toán phân loại hoa Iris")
    st.write(
        "Bộ dữ liệu Iris được dùng để minh họa bài toán phân loại. "
        "Ứng dụng huấn luyện nhiều mô hình và so sánh độ chính xác trên tập kiểm thử."
    )

    score_df = pd.DataFrame({
        "Thuật toán": list(scores.keys()),
        "Accuracy (%)": [round(v * 100, 2) for v in scores.values()]
    })

    st.dataframe(score_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(score_df["Thuật toán"], score_df["Accuracy (%)"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticklabels(score_df["Thuật toán"], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with demo_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Thử dự đoán loài hoa")

    sepal_length = st.slider("Chiều dài đài hoa", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Chiều rộng đài hoa", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Chiều dài cánh hoa", 1.0, 7.0, 4.3, 0.1)
    petal_width = st.slider("Chiều rộng cánh hoa", 0.1, 2.5, 1.3, 0.1)

    if st.button("🚀 Dự đoán bằng mô hình tốt nhất"):
        sample = pd.DataFrame([{
            "Chiều dài đài hoa": sepal_length,
            "Chiều rộng đài hoa": sepal_width,
            "Chiều dài cánh hoa": petal_length,
            "Chiều rộng cánh hoa": petal_width
        }])

        result = best_model.predict(sample)[0]
        proba = best_model.predict_proba(sample).max() * 100

        st.markdown(f"""
        <div class="result-good">
            <h2>Kết quả dự đoán</h2>
            <h1>{target_names[result].title()} ✅</h1>
            <p>Mô hình tốt nhất: <b>{best_name}</b></p>
            <p>Độ tin cậy: <b>{proba:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <b>Hướng dẫn:</b><br>
            Điều chỉnh các thông số của hoa, sau đó bấm nút dự đoán để xem kết quả.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ===================== CONFUSION MATRIX + TABLE =====================
st.markdown('<div class="section-title">5. Đánh giá mô hình và bảng dữ liệu</div>', unsafe_allow_html=True)

cm_col, table_col = st.columns([1, 1.15])

with cm_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"### 🔍 Ma trận nhầm lẫn: {best_name}")
    cm = confusion_matrix(y_test, predictions[best_name])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cm)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(target_names, rotation=25, ha="right")
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with table_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Bảng so sánh thuật toán")
    st.dataframe(algorithms, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


with st.expander("📄 Xem dữ liệu Iris dùng trong demo"):
    iris_df = X.copy()
    iris_df["Loài hoa"] = [target_names[i] for i in y]
    st.dataframe(iris_df, use_container_width=True)


st.markdown("""
<br>
<div style="text-align:center;color:#cbd5e1;padding:20px;">
    Made with ❤️ using Python, Streamlit & Scikit-learn<br>
    Đề tài: Tìm hiểu về một số thuật toán học máy và ứng dụng trong cuộc sống
</div>
""", unsafe_allow_html=True)
