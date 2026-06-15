import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="Student Performance ML App",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #172554 0, #020617 35%, #020617 100%);
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #f8fafc;
}
.hero {
    padding: 34px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(59,130,246,0.35), rgba(147,51,234,0.32));
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}
.hero h1 {
    font-size: 42px;
    line-height: 1.2;
    margin-bottom: 16px;
}
.hero p {
    font-size: 18px;
    color: #dbeafe;
}
.card {
    padding: 22px;
    border-radius: 18px;
    background: rgba(15,23,42,0.82);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 12px 35px rgba(0,0,0,0.25);
}
.metric-number {
    font-size: 34px;
    font-weight: 800;
    color: #ffffff;
}
.metric-label {
    font-size: 14px;
    color: #cbd5e1;
}
.section-title {
    font-size: 22px;
    font-weight: 800;
    margin-top: 22px;
    margin-bottom: 12px;
}
.result-good {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(34,197,94,0.24), rgba(16,185,129,0.14));
    border: 1px solid rgba(74,222,128,0.45);
    text-align: center;
}
.result-bad {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(239,68,68,0.24), rgba(244,63,94,0.14));
    border: 1px solid rgba(248,113,113,0.45);
    text-align: center;
}
div.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: 0;
    padding: 13px 18px;
    font-weight: 800;
    color: white;
    background: linear-gradient(90deg, #2563eb, #9333ea);
}
div.stButton > button:hover {
    color: white;
    border: 0;
    filter: brightness(1.12);
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA ----------------
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "Toán": np.random.normal(14, 3, n).clip(0, 20).round(1),
    "Anh văn": np.random.normal(13.5, 3.2, n).clip(0, 20).round(1),
    "Tin học": np.random.normal(14.5, 2.8, n).clip(0, 20).round(1),
    "Tự học/tuần": np.random.randint(1, 26, n),
    "Vắng học": np.random.randint(0, 12, n),
    "Hoạt động ngoại khóa": np.random.choice([0, 1], n, p=[0.45, 0.55])
})

score = (
    df["Toán"] * 0.28 +
    df["Anh văn"] * 0.18 +
    df["Tin học"] * 0.28 +
    df["Tự học/tuần"] * 0.18 -
    df["Vắng học"] * 0.45 +
    df["Hoạt động ngoại khóa"] * 0.8
)

df["Kết quả"] = np.where(score >= 11.5, "Đạt", "Chưa đạt")

X = df[["Toán", "Anh văn", "Tin học", "Tự học/tuần", "Vắng học", "Hoạt động ngoại khóa"]]
y = df["Kết quả"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, pos_label="Đạt")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎓 Student Performance")
    st.caption("Machine Learning App")
    st.markdown("---")
    st.markdown("### KHÁM PHÁ")
    st.markdown("🏠 Trang chủ")
    st.markdown("📊 Phân tích dữ liệu")
    st.markdown("🤖 Mô hình & đánh giá")
    st.markdown("🎯 Dự đoán kết quả")
    st.markdown("---")
    st.markdown("### THÔNG TIN")
    st.info("Ứng dụng sử dụng Random Forest để dự đoán kết quả học tập sinh viên dựa trên điểm số và thói quen học tập.")

# ---------------- HERO ----------------
left, right = st.columns([2.2, 1])

with left:
    st.markdown("""
    <div class="hero">
        <h1>Dự đoán kết quả học tập sinh viên 🚀</h1>
        <p>
        Ứng dụng Machine Learning sử dụng dữ liệu học tập để phân tích,
        trực quan hóa và dự đoán khả năng đạt/chưa đạt của sinh viên.
        </p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Dự đoán nhanh")
    math = st.slider("Điểm Toán", 0.0, 20.0, 15.0, 0.5)
    english = st.slider("Điểm Anh văn", 0.0, 20.0, 14.0, 0.5)
    it = st.slider("Điểm Tin học", 0.0, 20.0, 15.0, 0.5)
    self_study = st.slider("Số giờ tự học/tuần", 0, 30, 12)
    absence = st.slider("Số buổi vắng học", 0, 15, 2)
    activity = st.selectbox("Tham gia ngoại khóa", ["Có", "Không"])
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- METRICS ----------------
st.markdown('<div class="section-title">Tổng quan ứng dụng</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)

metrics = [
    ("1000", "Dòng dữ liệu sinh viên", "🗃️"),
    ("5", "Biểu đồ trực quan", "📊"),
    (f"{accuracy*100:.1f}%", "Độ chính xác mô hình", "✅"),
    ("Random Forest", "Thuật toán tốt nhất", "🌲")
]

for col, (num, label, icon) in zip([m1, m2, m3, m4], metrics):
    with col:
        st.markdown(f"""
        <div class="card">
            <div style="font-size:30px">{icon}</div>
            <div class="metric-number">{num}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- CHARTS ----------------
st.markdown('<div class="section-title">Phân tích dữ liệu khám phá</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Phân bố kết quả")
    counts = df["Kết quả"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Điểm trung bình")
    means = df[["Toán", "Anh văn", "Tin học"]].mean()
    fig, ax = plt.subplots()
    ax.bar(means.index, means.values)
    ax.set_ylim(0, 20)
    ax.set_ylabel("Điểm")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Tương quan dữ liệu")
    corr = df[["Toán", "Anh văn", "Tin học", "Tự học/tuần", "Vắng học"]].corr()
    fig, ax = plt.subplots()
    im = ax.imshow(corr)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.tick_params(colors="white")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MODEL + PREDICTION ----------------
st.markdown('<div class="section-title">Mô hình Machine Learning & Kết quả dự đoán</div>', unsafe_allow_html=True)
model_col, predict_col = st.columns([1.4, 1])

with model_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Đánh giá mô hình")
    cm = confusion_matrix(y_test, pred, labels=["Đạt", "Chưa đạt"])
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Đạt", "Chưa đạt"])
    ax.set_yticklabels(["Đạt", "Chưa đạt"])
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=16)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    st.pyplot(fig)
    st.write(f"**Accuracy:** {accuracy*100:.2f}%")
    st.write(f"**F1-score:** {f1:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

with predict_col:
    input_data = pd.DataFrame([{
        "Toán": math,
        "Anh văn": english,
        "Tin học": it,
        "Tự học/tuần": self_study,
        "Vắng học": absence,
        "Hoạt động ngoại khóa": 1 if activity == "Có" else 0
    }])

    if st.button("🚀 Dự đoán kết quả"):
        result = model.predict(input_data)[0]
        proba = model.predict_proba(input_data).max() * 100

        if result == "Đạt":
            st.markdown(f"""
            <div class="result-good">
                <h1>ĐẠT ✅</h1>
                <p>Độ tin cậy dự đoán: <b>{proba:.1f}%</b></p>
                <p>Sinh viên có khả năng đạt kết quả học tập tốt.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-bad">
                <h1>CHƯA ĐẠT ⚠️</h1>
                <p>Độ tin cậy dự đoán: <b>{proba:.1f}%</b></p>
                <p>Cần tăng thời gian tự học và giảm số buổi vắng học.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card">
            <h3>📌 Hướng dẫn</h3>
            <p>Điều chỉnh thông tin ở khung bên phải phía trên, sau đó bấm nút dự đoán.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- TABLE ----------------
with st.expander("📄 Xem một phần dữ liệu mẫu"):
    st.dataframe(df.head(30), use_container_width=True)

st.markdown("""
<br>
<div style="text-align:center;color:#94a3b8">
Made with ❤️ using Streamlit & Scikit-learn<br>
© Student Performance App | Nhập môn Khoa học dữ liệu
</div>
""", unsafe_allow_html=True)
