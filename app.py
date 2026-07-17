# -*- coding: utf-8 -*-
"""
เว็บแอปทำนายภาวะซึมเศร้าด้วยโมเดล SVM
โครงสร้างแบบเดียวกับโปรเจกต์ Iris: โหลด scaler.pkl + model.pkl แยกกัน
วิธีรัน:  streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np

# =========================
# Load Model (แบบเดียวกับโปรเจกต์ Iris)
# =========================
scaler = joblib.load("scaler.pkl")
model = joblib.load("svm_depression_model.pkl")

# =========================
# Page
# =========================
st.set_page_config(
    page_title="MindCheck | ระบบทำนายภาวะซึมเศร้า",
    page_icon="🌿",
    layout="centered"
)

# =========================
# CSS ตกแต่ง — ธีม "ใบไม้ยามเช้า" + ฟอนต์ไทย Mitr/Anuphan
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Mitr:wght@500;600&family=Anuphan:wght@400;500;600&display=swap');

.stApp {
    background: linear-gradient(180deg, #EEF5F0 0%, #F7FAF8 40%);
    font-family: 'Anuphan', sans-serif;
}
html, body, [class*="css"], p, label, span, div {
    font-family: 'Anuphan', sans-serif;
}
h1, h2, h3 {
    font-family: 'Mitr', sans-serif !important;
    color: #17453B;
}

/* การ์ดส่วนหัว */
.hero {
    background: linear-gradient(135deg, #17453B 0%, #2E7D66 100%);
    border-radius: 20px;
    padding: 2.2rem 2rem 1.8rem 2rem;
    margin-bottom: 1.2rem;
    color: #FFFFFF;
    box-shadow: 0 8px 24px rgba(23, 69, 59, 0.25);
}
.hero h1 { color: #FFFFFF !important; font-size: 1.9rem; margin: 0 0 0.4rem 0; }
.hero p { color: #CDE8DD; margin: 0; font-size: 1rem; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 999px;
    padding: 0.15rem 0.8rem;
    font-size: 0.8rem;
    margin-top: 0.8rem;
    color: #EAF6F0;
}

/* กล่องคำเตือน */
.notice {
    background: #FFF8E8;
    border-left: 5px solid #E3A82B;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    font-size: 0.92rem;
    color: #6B5312;
    margin-bottom: 1.4rem;
}

/* หัวข้อกลุ่มฟอร์ม */
.section-label {
    font-family: 'Mitr', sans-serif;
    color: #2E7D66;
    font-size: 1.05rem;
    border-bottom: 2px solid #CDE8DD;
    padding-bottom: 0.3rem;
    margin: 1.2rem 0 0.6rem 0;
}

/* ปุ่มทำนาย */
.stButton > button {
    background: linear-gradient(135deg, #2E7D66 0%, #17453B 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1rem;
    font-family: 'Mitr', sans-serif;
    font-size: 1.1rem;
    box-shadow: 0 4px 14px rgba(46, 125, 102, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(46, 125, 102, 0.45);
    color: #FFFFFF;
}

/* การ์ดผลลัพธ์ */
.result-card {
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-top: 1rem;
    animation: fadeUp 0.5s ease;
}
.result-safe { background: #E9F7EE; border: 1.5px solid #7BC79A; color: #1D5B34; }
.result-risk { background: #FDEEEC; border: 1.5px solid #E58B7B; color: #7A2C1E; }
.result-card h2 { font-size: 1.35rem; margin: 0 0 0.4rem 0; }
.result-safe h2 { color: #1D5B34 !important; }
.result-risk h2 { color: #7A2C1E !important; }
.result-card p { margin: 0.2rem 0; font-size: 0.95rem; }

/* เกจความเสี่ยง */
.gauge-wrap {
    background: #E4EEE8;
    border-radius: 999px;
    height: 18px;
    overflow: hidden;
    margin: 0.8rem 0 0.3rem 0;
}
.gauge-fill { height: 100%; border-radius: 999px; transition: width 0.8s ease; }
.gauge-caption { font-size: 0.85rem; color: #4A6B5D; text-align: right; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.footer-note {
    text-align: center;
    color: #7C9C8E;
    font-size: 0.82rem;
    margin-top: 2.2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# ส่วนหัวเว็บ
# =========================
st.markdown("""
<div class="hero">
    <h1>🌿 MindCheck — ระบบทำนายภาวะซึมเศร้า</h1>
    <p>ประเมินความเสี่ยงเบื้องต้นจากพฤติกรรมและไลฟ์สไตล์ ด้วยโมเดล Support Vector Machine</p>
    <span class="badge">SVM · Accuracy 93.6% · F1-score 0.82</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
⚠️ ระบบนี้เป็นโปรเจกต์เพื่อการศึกษา <b>ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์</b>
หากมีความกังวลเรื่องสุขภาพจิต ปรึกษาสายด่วนสุขภาพจิต <b>โทร 1323</b> (ฟรี ตลอด 24 ชม.)
</div>
""", unsafe_allow_html=True)

# =========================
# Input — แบ่งเป็น 3 กลุ่มให้อ่านง่าย
# =========================
st.markdown('<div class="section-label">👤 ข้อมูลทั่วไป</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("เพศ", ["Male", "Female"],
                          format_func=lambda x: "ชาย" if x == "Male" else "หญิง")
with col2:
    age = st.slider("อายุ (ปี)", 18, 60, 25)
with col3:
    status = st.selectbox("สถานะ", ["Student", "Working Professional"],
                          format_func=lambda x: "นักศึกษา" if x == "Student" else "คนทำงาน")

st.markdown('<div class="section-label">📚 การเรียน / การทำงาน</div>', unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    pressure = st.slider("ความกดดันจากการเรียน/งาน (1=น้อย, 5=มาก)", 1.0, 5.0, 3.0, 0.5)
    satisfaction = st.slider("ความพึงพอใจในการเรียน/งาน (1=น้อย, 5=มาก)", 1.0, 5.0, 3.0, 0.5)
with col5:
    work_hours = st.slider("ชั่วโมงทำงาน/เรียนต่อวัน", 0.0, 16.0, 8.0, 0.5)
    cgpa = st.slider("เกรดเฉลี่ย CGPA (เฉพาะนักศึกษา)", 0.0, 10.0, 7.5, 0.1,
                     help="ถ้าเป็นคนทำงาน ปล่อยค่าเริ่มต้นไว้ได้")

st.markdown('<div class="section-label">💚 สุขภาพและไลฟ์สไตล์</div>', unsafe_allow_html=True)
col6, col7 = st.columns(2)
with col6:
    sleep = st.slider("ชั่วโมงนอนต่อวัน", 1.0, 10.0, 7.0, 0.5)
    diet = st.selectbox("พฤติกรรมการกิน", ["Healthy", "Moderate", "Unhealthy"],
                        format_func=lambda x: {"Healthy": "ดีต่อสุขภาพ",
                                               "Moderate": "ปานกลาง",
                                               "Unhealthy": "ไม่ดีต่อสุขภาพ"}[x])
    fin_stress = st.slider("ความเครียดด้านการเงิน (1=น้อย, 5=มาก)", 1.0, 5.0, 3.0, 0.5)
with col7:
    suicidal = st.selectbox("เคยมีความคิดทำร้ายตัวเองหรือไม่", ["No", "Yes"],
                            format_func=lambda x: "ไม่เคย" if x == "No" else "เคย")
    family = st.selectbox("ครอบครัวมีประวัติปัญหาสุขภาพจิตหรือไม่", ["No", "Yes"],
                          format_func=lambda x: "ไม่มี" if x == "No" else "มี")

st.write("")

# =========================
# Prediction (แบบเดียวกับโปรเจกต์ Iris: numpy array -> scale -> predict)
# =========================
if st.button("🔍 ประเมินความเสี่ยง", type="primary", use_container_width=True):

    # แปลงข้อความเป็นตัวเลข "ต้องตรงกับตอนเทรน" ทุกตัว
    GENDER_MAP = {"Male": 0, "Female": 1}
    STATUS_MAP = {"Student": 0, "Working Professional": 1}
    DIET_MAP = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}
    YESNO_MAP = {"No": 0, "Yes": 1}

    # ลำดับ Feature 12 ตัว ต้องเรียงเหมือนตอนเทรนเป๊ะๆ:
    # [Gender, Age, Status, Pressure, CGPA, Satisfaction,
    #  Sleep Hours, Diet, Suicidal, Work/Study Hours, Financial Stress, Family History]
    input_data = np.array([[
        GENDER_MAP[gender],
        age,
        STATUS_MAP[status],
        pressure,
        cgpa,
        satisfaction,
        sleep,
        DIET_MAP[diet],
        YESNO_MAP[suicidal],
        work_hours,
        fin_stress,
        YESNO_MAP[family],
    ]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Probability (ถ้ามี)
    try:
        risk = float(model.predict_proba(input_scaled)[0][1])
    except Exception:
        risk = None

    # เลือกสีเกจตามระดับความเสี่ยง
    if risk is not None:
        gauge_color = "#7BC79A" if risk < 0.35 else ("#E3A82B" if risk < 0.65 else "#D96C57")
        gauge_html = f"""
            <div class="gauge-wrap"><div class="gauge-fill" style="width:{risk*100:.0f}%; background:{gauge_color};"></div></div>
            <div class="gauge-caption">ระดับความเสี่ยง {risk:.1%}</div>"""
        risk_text = f"โมเดลประเมินความน่าจะเป็นอยู่ที่ <b>{risk:.1%}</b>"
    else:
        gauge_html = ""
        risk_text = ""

    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-risk">
            <h2>ผลประเมิน: มีความเสี่ยงภาวะซึมเศร้า</h2>
            <p>{risk_text}</p>
            {gauge_html}
            <p style="margin-top:0.8rem;">ผลนี้เป็นการประเมินเบื้องต้นจากโมเดลเท่านั้น
            แนะนำให้พูดคุยกับคนใกล้ชิดหรือผู้เชี่ยวชาญด้านสุขภาพจิต<br>
            📞 สายด่วนสุขภาพจิต <b>1323</b> (ฟรี ตลอด 24 ชั่วโมง)</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-safe">
            <h2>ผลประเมิน: ไม่พบความเสี่ยงภาวะซึมเศร้า</h2>
            <p>{risk_text}</p>
            {gauge_html}
            <p style="margin-top:0.8rem;">อย่าลืมดูแลสุขภาพกายและใจอย่างสม่ำเสมอนะครับ 🌱</p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# ส่วนท้าย
# =========================
st.markdown("""
<div class="footer-note">
โปรเจกต์เพื่อการศึกษา · Machine Learning : Support Vector Machine (SVM)
</div>
""", unsafe_allow_html=True)
