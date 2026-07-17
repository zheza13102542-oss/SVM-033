# -*- coding: utf-8 -*-
"""
=====================================================================
 เว็บแอปทำนายภาวะซึมเศร้าด้วยโมเดล SVM (MindCheck)
=====================================================================
 หลักการทำงานของแอปนี้ (เหมือนโปรเจกต์ Iris เดิม):

   ผู้ใช้กรอกฟอร์ม 12 ช่อง
        ↓
   แปลงคำตอบที่เป็นข้อความให้เป็นตัวเลข (Encoding)
        ↓
   จัดเรียงเป็น numpy array 1 แถว 12 คอลัมน์
        ↓
   ปรับสเกลด้วย scaler.pkl (StandardScaler ตัวเดียวกับตอนเทรน)
        ↓
   ส่งเข้าโมเดล svm_depression_model.pkl เพื่อทำนาย
        ↓
   แสดงผล 0 = ไม่มีความเสี่ยง / 1 = มีความเสี่ยง พร้อม % ความน่าจะเป็น

 ไฟล์ที่ต้องอยู่ในโฟลเดอร์เดียวกัน:
   - app.py (ไฟล์นี้)
   - scaler.pkl                  <- ตัวปรับสเกลข้อมูล
   - svm_depression_model.pkl    <- โมเดล SVM ที่เทรนแล้ว

 วิธีรัน:  streamlit run app.py
=====================================================================
"""

# =====================================================================
# ส่วนที่ 1: Import ไลบรารีที่จำเป็น
# =====================================================================
import streamlit as st   # สร้างหน้าเว็บ (ฟอร์ม, ปุ่ม, แสดงผล)
import joblib            # โหลดไฟล์ .pkl ที่บันทึกไว้ตอนเทรน
import numpy as np       # สร้าง array ข้อมูลก่อนส่งเข้าโมเดล

# =====================================================================
# ส่วนที่ 2: โหลด Scaler และโมเดลที่เทรนไว้แล้ว
# =====================================================================
# ต้องโหลดทั้ง 2 ไฟล์ เพราะตอนเทรนเราแยกบันทึกไว้:
#   - scaler.pkl : จำค่าเฉลี่ย (mean) และส่วนเบี่ยงเบน (std) ของชุด train
#                  เอาไว้ปรับสเกลข้อมูลใหม่ให้อยู่ในมาตรฐานเดียวกับตอนเทรน
#   - svm_depression_model.pkl : โมเดล SVM ที่หาเส้นแบ่งไว้เรียบร้อยแล้ว
# ข้อควรรู้: SVM ต้องรับข้อมูลที่ผ่านการ scale เสมอ ถ้าลืม scale ผลทำนายจะผิดหมด
scaler = joblib.load("scaler.pkl")
model = joblib.load("svm_depression_model.pkl")

# =====================================================================
# ส่วนที่ 3: ตารางแปลงข้อความเป็นตัวเลข (Encoding Map)
# =====================================================================
# โมเดลรับได้เฉพาะ "ตัวเลข" เท่านั้น รับข้อความอย่าง "Male" ตรงๆ ไม่ได้
# ตารางนี้ต้องตรงกับที่ใช้ตอนเทรนใน train_svm.py แบบเป๊ะๆ ทุกตัว
# (ถ้าตอนเทรน Male=0 แต่ในแอปใส่ Male=1 โมเดลจะทำนายผิดทันทีโดยไม่มี error เตือน)
GENDER_MAP = {"Male": 0, "Female": 1}                    # เพศ
STATUS_MAP = {"Student": 0, "Working Professional": 1}   # สถานะ
DIET_MAP = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2} # พฤติกรรมการกิน
YESNO_MAP = {"No": 0, "Yes": 1}                          # คำถามตอบ ใช่/ไม่ใช่

# =====================================================================
# ส่วนที่ 4: ตั้งค่าพื้นฐานของหน้าเว็บ
# =====================================================================
# page_title = ข้อความบนแท็บเบราว์เซอร์ | page_icon = ไอคอนแท็บ
# layout="centered" = จัดเนื้อหาอยู่กลางจอ อ่านง่ายบนทุกขนาดหน้าจอ
st.set_page_config(
    page_title="MindCheck | ระบบทำนายภาวะซึมเศร้า",
    page_icon="🌿",
    layout="centered"
)

# =====================================================================
# ส่วนที่ 5: CSS ตกแต่งหน้าเว็บ — ธีม "ใบไม้ยามเช้า"
# =====================================================================
# Streamlit ให้เราแทรก CSS เองได้ผ่าน st.markdown + unsafe_allow_html=True
# สิ่งที่ CSS ชุดนี้ทำ:
#   1. โหลดฟอนต์ไทยจาก Google Fonts (Mitr = หัวข้อ, Anuphan = เนื้อหา)
#   2. พื้นหลังไล่เฉดเขียวอ่อน ให้ความรู้สึกสงบ เหมาะกับแอปสุขภาพจิต
#   3. .hero      = การ์ดส่วนหัวสีเขียวเข้ม
#   4. .notice    = กล่องคำเตือนสีเหลือง
#   5. .stButton  = ปุ่มทำนาย มี hover effect ยกตัวขึ้น
#   6. .result-*  = การ์ดแสดงผล (เขียว=ปลอดภัย / แดง=เสี่ยง) + แอนิเมชัน fadeUp
#   7. .gauge-*   = แถบเกจแสดงระดับความเสี่ยงเป็น %
st.markdown("""
<style>
/* โหลดฟอนต์ไทยจาก Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Mitr:wght@500;600&family=Anuphan:wght@400;500;600&display=swap');

/* พื้นหลังหลักของแอป: ไล่เฉดจากเขียวอ่อนลงมาขาวนวล */
.stApp {
    background: linear-gradient(180deg, #EEF5F0 0%, #F7FAF8 40%);
    font-family: 'Anuphan', sans-serif;
}
/* บังคับให้ทุกข้อความในหน้าใช้ฟอนต์ Anuphan */
html, body, [class*="css"], p, label, span, div {
    font-family: 'Anuphan', sans-serif;
}
/* หัวข้อทุกระดับใช้ฟอนต์ Mitr สีเขียวเข้ม */
h1, h2, h3 {
    font-family: 'Mitr', sans-serif !important;
    color: #17453B;
}

/* การ์ดส่วนหัว (hero): พื้นไล่เฉดเขียวเข้ม มุมโค้ง มีเงา */
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
/* ป้าย badge เล็กๆ บอกสเปคโมเดล ใต้หัวข้อ */
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

/* กล่องคำเตือน: พื้นเหลืองอ่อน มีแถบสีเข้มด้านซ้าย */
.notice {
    background: #FFF8E8;
    border-left: 5px solid #E3A82B;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    font-size: 0.92rem;
    color: #6B5312;
    margin-bottom: 1.4rem;
}

/* หัวข้อของแต่ละกลุ่มฟอร์ม: ฟอนต์ Mitr + เส้นใต้สีเขียวอ่อน */
.section-label {
    font-family: 'Mitr', sans-serif;
    color: #2E7D66;
    font-size: 1.05rem;
    border-bottom: 2px solid #CDE8DD;
    padding-bottom: 0.3rem;
    margin: 1.2rem 0 0.6rem 0;
}

/* ปุ่มทำนาย: พื้นไล่เฉดเขียว มุมโค้ง มีเงา */
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
/* ตอนเอาเมาส์ชี้ปุ่ม: ยกตัวขึ้น 2px และเงาเข้มขึ้น */
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(46, 125, 102, 0.45);
    color: #FFFFFF;
}

/* การ์ดแสดงผลลัพธ์: มุมโค้ง + แอนิเมชันลอยขึ้นตอนปรากฏ */
.result-card {
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-top: 1rem;
    animation: fadeUp 0.5s ease;
}
/* โทนเขียว = ไม่พบความเสี่ยง */
.result-safe { background: #E9F7EE; border: 1.5px solid #7BC79A; color: #1D5B34; }
/* โทนแดงส้ม = มีความเสี่ยง */
.result-risk { background: #FDEEEC; border: 1.5px solid #E58B7B; color: #7A2C1E; }
.result-card h2 { font-size: 1.35rem; margin: 0 0 0.4rem 0; }
.result-safe h2 { color: #1D5B34 !important; }
.result-risk h2 { color: #7A2C1E !important; }
.result-card p { margin: 0.2rem 0; font-size: 0.95rem; }

/* เกจความเสี่ยง: แถบพื้น (gauge-wrap) + แถบสีที่ยาวตาม % (gauge-fill) */
.gauge-wrap {
    background: #E4EEE8;
    border-radius: 999px;
    height: 18px;
    overflow: hidden;
    margin: 0.8rem 0 0.3rem 0;
}
.gauge-fill { height: 100%; border-radius: 999px; transition: width 0.8s ease; }
.gauge-caption { font-size: 0.85rem; color: #4A6B5D; text-align: right; }

/* แอนิเมชัน: เลื่อนขึ้น 10px พร้อมค่อยๆ ชัดขึ้น ใน 0.5 วินาที */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ข้อความส่วนท้ายหน้า */
.footer-note {
    text-align: center;
    color: #7C9C8E;
    font-size: 0.82rem;
    margin-top: 2.2rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# ส่วนที่ 6: ส่วนหัวของหน้าเว็บ (Hero + คำเตือน)
# =====================================================================
st.markdown("""
<div class="hero">
    <h1>🌿 MindCheck — ระบบทำนายภาวะซึมเศร้า</h1>
    <p>ประเมินความเสี่ยงเบื้องต้นจากพฤติกรรมและไลฟ์สไตล์ ด้วยโมเดล Support Vector Machine</p>
    <span class="badge">SVM · Accuracy 93.6% · F1-score 0.82</span>
</div>
""", unsafe_allow_html=True)

# กล่องคำเตือน: จำเป็นต้องมีตามหลักจริยธรรม เพราะเป็นระบบเกี่ยวกับสุขภาพจิต
st.markdown("""
<div class="notice">
⚠️ ระบบนี้เป็นโปรเจกต์เพื่อการศึกษา <b>ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์</b>
หากมีความกังวลเรื่องสุขภาพจิต ปรึกษาสายด่วนสุขภาพจิต <b>โทร 1323</b> (ฟรี ตลอด 24 ชม.)
</div>
""", unsafe_allow_html=True)

# =====================================================================
# ส่วนที่ 7: ฟอร์มกรอกข้อมูล 12 ช่อง (แบ่ง 3 กลุ่มให้กรอกง่าย)
# =====================================================================
# เทคนิคที่ใช้:
#   - st.columns(n)  = แบ่งหน้าเป็น n คอลัมน์ วางช่องกรอกเคียงกัน
#   - st.selectbox   = เมนูเลือกตัวเลือก / st.slider = แถบเลื่อนตัวเลข
#   - format_func    = "แสดง" เป็นภาษาไทยให้ผู้ใช้อ่าน แต่ "ค่าจริง" ในตัวแปร
#                      ยังเป็นภาษาอังกฤษตรงกับ Encoding Map (เช่น เห็น "ชาย"
#                      แต่ตัวแปร gender เก็บค่า "Male")

# ---- กลุ่มที่ 1: ข้อมูลทั่วไป (3 ช่อง) ----
st.markdown('<div class="section-label">👤 ข้อมูลทั่วไป</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("เพศ", ["Male", "Female"],
                          format_func=lambda x: "ชาย" if x == "Male" else "หญิง")
with col2:
    age = st.slider("อายุ (ปี)", 18, 60, 25)   # ต่ำสุด 18, สูงสุด 60, ค่าเริ่มต้น 25
with col3:
    status = st.selectbox("สถานะ", ["Student", "Working Professional"],
                          format_func=lambda x: "นักศึกษา" if x == "Student" else "คนทำงาน")

# ---- กลุ่มที่ 2: การเรียน/การทำงาน (4 ช่อง) ----
st.markdown('<div class="section-label">📚 การเรียน / การทำงาน</div>', unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    # สเกล 1-5 ตามชุดข้อมูลจริง (1 = น้อยสุด, 5 = มากสุด)
    pressure = st.slider("ความกดดันจากการเรียน/งาน (1=น้อย, 5=มาก)", 1.0, 5.0, 3.0, 0.5)
    satisfaction = st.slider("ความพึงพอใจในการเรียน/งาน (1=น้อย, 5=มาก)", 1.0, 5.0, 3.0, 0.5)
with col5:
    work_hours = st.slider("ชั่วโมงทำงาน/เรียนต่อวัน", 0.0, 16.0, 8.0, 0.5)
    # CGPA ในชุดข้อมูลนี้เป็นสเกล 0-10 (ระบบอินเดีย) ไม่ใช่ 0-4 แบบไทย
    cgpa = st.slider("เกรดเฉลี่ย CGPA (เฉพาะนักศึกษา)", 0.0, 10.0, 7.5, 0.1,
                     help="ถ้าเป็นคนทำงาน ปล่อยค่าเริ่มต้นไว้ได้")

# ---- กลุ่มที่ 3: สุขภาพและไลฟ์สไตล์ (5 ช่อง) ----
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

st.write("")   # เว้นบรรทัดว่างก่อนปุ่ม

# =====================================================================
# ส่วนที่ 8: กดปุ่มแล้วทำนาย
# =====================================================================
# โค้ดใน if นี้จะทำงาน "เฉพาะตอนผู้ใช้กดปุ่ม" เท่านั้น
if st.button("🔍 ประเมินความเสี่ยง", type="primary", use_container_width=True):

    # ---- 8.1 แปลงคำตอบเป็นตัวเลข แล้วเรียงเป็น array 1 แถว 12 คอลัมน์ ----
    # !! สำคัญที่สุดในไฟล์นี้ !!
    # ลำดับทั้ง 12 ตัวต้องเรียงตรงกับตอนเทรนใน train_svm.py เป๊ะๆ:
    #   ตำแหน่ง 0: Gender          (0=ชาย, 1=หญิง)
    #   ตำแหน่ง 1: Age             (อายุ)
    #   ตำแหน่ง 2: Status          (0=นักศึกษา, 1=คนทำงาน)
    #   ตำแหน่ง 3: Pressure        (ความกดดัน 1-5)
    #   ตำแหน่ง 4: CGPA            (เกรดเฉลี่ย 0-10)
    #   ตำแหน่ง 5: Satisfaction    (ความพึงพอใจ 1-5)
    #   ตำแหน่ง 6: Sleep Hours     (ชั่วโมงนอน)
    #   ตำแหน่ง 7: Diet            (0=ดี, 1=กลาง, 2=ไม่ดี)
    #   ตำแหน่ง 8: Suicidal        (0=ไม่เคย, 1=เคย)
    #   ตำแหน่ง 9: Work/Study Hours (ชั่วโมงทำงาน/เรียน)
    #   ตำแหน่ง 10: Financial Stress (ความเครียดการเงิน 1-5)
    #   ตำแหน่ง 11: Family History  (0=ไม่มี, 1=มี)
    input_data = np.array([[
        GENDER_MAP[gender],      # 0: เพศ
        age,                     # 1: อายุ
        STATUS_MAP[status],      # 2: สถานะ
        pressure,                # 3: ความกดดัน
        cgpa,                    # 4: เกรดเฉลี่ย
        satisfaction,            # 5: ความพึงพอใจ
        sleep,                   # 6: ชั่วโมงนอน
        DIET_MAP[diet],          # 7: การกิน
        YESNO_MAP[suicidal],     # 8: ความคิดทำร้ายตัวเอง
        work_hours,              # 9: ชั่วโมงทำงาน/เรียน
        fin_stress,              # 10: ความเครียดการเงิน
        YESNO_MAP[family],       # 11: ประวัติครอบครัว
    ]])

    # ---- 8.2 ปรับสเกลด้วย scaler ตัวเดียวกับตอนเทรน ----
    # ใช้ .transform() เท่านั้น (ห้ามใช้ .fit ในแอป เพราะจะไปคำนวณ
    # mean/std ใหม่จากข้อมูลแค่ 1 แถว ทำให้สเกลเพี้ยนจากตอนเทรน)
    input_scaled = scaler.transform(input_data)

    # ---- 8.3 ทำนายผล ----
    # .predict() คืนค่า 0 หรือ 1
    prediction = model.predict(input_scaled)[0]

    # ---- 8.4 ขอค่าความน่าจะเป็น (ถ้าโมเดลรองรับ) ----
    # predict_proba คืนค่า [P(class 0), P(class 1)] เช่น [0.85, 0.15]
    # เราสนใจตัวหลัง (index 1) = ความน่าจะเป็นที่จะ "มีความเสี่ยง"
    # ใส่ try/except กันพัง เผื่อโมเดลบางแบบไม่มี predict_proba
    try:
        risk = float(model.predict_proba(input_scaled)[0][1])
    except Exception:
        risk = None

    # ---- 8.5 เตรียม HTML แสดงเกจความเสี่ยง ----
    # สีของเกจเปลี่ยนตามระดับ: เขียว (<35%) เหลือง (35-65%) แดง (>65%)
    if risk is not None:
        gauge_color = "#7BC79A" if risk < 0.35 else ("#E3A82B" if risk < 0.65 else "#D96C57")
        gauge_html = f"""
            <div class="gauge-wrap"><div class="gauge-fill" style="width:{risk*100:.0f}%; background:{gauge_color};"></div></div>
            <div class="gauge-caption">ระดับความเสี่ยง {risk:.1%}</div>"""
        risk_text = f"โมเดลประเมินความน่าจะเป็นอยู่ที่ <b>{risk:.1%}</b>"
    else:
        gauge_html = ""
        risk_text = ""

    # ---- 8.6 แสดงการ์ดผลลัพธ์ตามคำทำนาย ----
    if prediction == 1:
        # ทำนายว่า "มีความเสี่ยง" -> การ์ดโทนแดง + ช่องทางขอความช่วยเหลือ
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
        # ทำนายว่า "ไม่มีความเสี่ยง" -> การ์ดโทนเขียว
        st.markdown(f"""
        <div class="result-card result-safe">
            <h2>ผลประเมิน: ไม่พบความเสี่ยงภาวะซึมเศร้า</h2>
            <p>{risk_text}</p>
            {gauge_html}
            <p style="margin-top:0.8rem;">อย่าลืมดูแลสุขภาพกายและใจอย่างสม่ำเสมอนะครับ 🌱</p>
        </div>
        """, unsafe_allow_html=True)

    # ---- 8.7 (เสริม) กดดูข้อมูลตัวเลขที่ส่งเข้าโมเดลจริงๆ ----
    # มีไว้ช่วยตอนสาธิต/ตอบคำถาม ว่าข้อมูลถูก encode เป็นตัวเลขอะไรบ้าง
    with st.expander("🔎 ดูข้อมูลที่ถูกแปลงเป็นตัวเลขก่อนส่งเข้าโมเดล"):
        feature_names = ["Gender", "Age", "Status", "Pressure", "CGPA",
                         "Satisfaction", "Sleep Hours", "Diet", "Suicidal",
                         "Work/Study Hours", "Financial Stress", "Family History"]
        st.write({name: val for name, val in zip(feature_names, input_data[0])})

# =====================================================================
# ส่วนที่ 9: ส่วนท้ายหน้าเว็บ
# =====================================================================
st.markdown("""
<div class="footer-note">
โปรเจกต์เพื่อการศึกษา · Machine Learning : Support Vector Machine (SVM)
</div>
""", unsafe_allow_html=True)