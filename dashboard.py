import streamlit as st
import pandas as pd
import joblib
from deepface import DeepFace
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# =========================
# 🌈 UI CONFIG
# =========================
st.set_page_config(page_title="AI Dashboard", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
.metric-card {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.5);
}
.success-box {
    background: #065f46;
    padding: 15px;
    border-radius: 10px;
    color: #4ade80;
}
.warning-box {
    background: #78350f;
    padding: 15px;
    border-radius: 10px;
    color: #facc15;
}
.section-card {
    background: #111827;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("AI Face Recognition Dashboard")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📊 Navigation")
option = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📸 Prediction", "📊 Analytics", "📁 Records"]
)

# LOAD MODEL
model = joblib.load("face_model.pkl")
# FILE
file = "attendance.xlsx"

# SAFE LOAD FUNCTION
def load_data():
    if os.path.exists(file):
        return pd.read_excel(file)
    return pd.DataFrame(columns=["Name", "Date", "Time"])

# ATTENDANCE FUNCTION

def mark_attendance(name):
    df_local = load_data()
    today = datetime.now().strftime("%Y-%m-%d")

    if not ((df_local["Name"] == name) & (df_local["Date"] == today)).any():
        new_row = {
            "Name": name,
            "Date": today,
            "Time": datetime.now().strftime("%H:%M:%S")
        }
        df_local = pd.concat([df_local, pd.DataFrame([new_row])], ignore_index=True)
        df_local.to_excel(file, index=False)

# =========================
# LOAD DATA
# =========================
df = load_data()
today = datetime.now().strftime("%Y-%m-%d")
today_df = df[df["Date"] == today]

# =========================
# METRICS
# =========================
dataset_path = "dataset/Faces"
all_people = os.listdir(dataset_path) if os.path.exists(dataset_path) else []

total_people = len(all_people)
present_today = today_df["Name"].nunique()
absent_today = max(0, total_people - present_today)

# =========================
# 🏠 HOME
# =========================
if option == "🏠 Home":

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""<div class="metric-card"><h2>{total_people}</h2><p>Total People</p></div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="metric-card"><h2>{present_today}</h2><p>Present Today</p></div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div class="metric-card"><h2>{absent_today}</h2><p>Absent</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    if present_today > 0:
        st.markdown('<div class="success-box">🟢 Attendance Running</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">🟡 No Attendance Yet</div>', unsafe_allow_html=True)

    st.subheader("✅ Present Today")

    if not today_df.empty:
        st.dataframe(today_df)
    else:
        st.info("No records yet")

    st.progress(present_today / total_people if total_people else 0)

# =========================
# 📸 PREDICTION
# =========================
elif option == "📸 Prediction":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    # 🔹 SINGLE IMAGE
    st.subheader("📸 Single Image")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image)

        temp_path = "temp_single.jpg"
        image.save(temp_path)

        try:
            result = DeepFace.represent(
                img_path=temp_path,
                model_name="Facenet",
                enforce_detection=False
            )

            embedding = result[0]["embedding"]
            prediction = model.predict([embedding])[0]

            st.success(f"👤 {prediction}")
            mark_attendance(prediction)

            os.remove(temp_path)
            st.rerun()

        except Exception as e:
            st.error(e)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🔹 BULK UPLOAD (FIXED)
# =========================
    st.subheader("📂 Bulk Upload")

uploaded_files = st.file_uploader(
    "Upload multiple images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:

    results = []

    for uploaded_file in uploaded_files:
        try:

            # ✅ FIX: use uploaded_file (NOT file)
            temp_path = f"temp_{uploaded_file.name}"

            # ✅ FIX: correct object
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            result = DeepFace.represent(
                img_path=temp_path,
                model_name="Facenet",
                enforce_detection=False
            )

            embedding = result[0]["embedding"]
            pred = model.predict([embedding])[0]

            mark_attendance(pred)

            results.append([uploaded_file.name, pred, "✅"])

            # ✅ Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            results.append([uploaded_file.name, str(e), "❌"])

    st.success("✅ Predictions Completed")

    df_results = pd.DataFrame(results, columns=["File", "Name", "Status"])
    st.dataframe(df_results)

# =========================
# 📊 ANALYTICS
# =========================
elif option == "📊 Analytics":

    if not df.empty:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Attendance Count")
            counts = df["Name"].value_counts()
            st.bar_chart(counts)

        with col2:
            st.subheader("📅 Daily Trend")
            daily = df.groupby("Date").size()
            st.line_chart(daily)

        st.subheader("🥧 Today Overview")

        fig, ax = plt.subplots()
        ax.pie(
            [present_today, absent_today],
            labels=["Present", "Absent"],
            autopct="%1.1f%%",
            colors=["#22c55e", "#ef4444"]
        )
        st.pyplot(fig)

    else:
        st.warning("No data available")

# =========================
# 📁 RECORDS
# =========================
elif option == "📁 Records":

    st.subheader("📅 Filter")
    date = st.date_input("Select Date")
    st.dataframe(df[df["Date"] == str(date)])

    st.subheader("🔍 Search")
    name = st.text_input("Enter name")

    if name:
        st.dataframe(df[df["Name"].str.contains(name, case=False)])

    st.subheader("📋 All Data")
    st.dataframe(df)

    st.download_button("⬇ Download CSV", df.to_csv(index=False), "attendance.csv")

    if st.button("🗑 Clear Data"):
        pd.DataFrame(columns=["Name","Date","Time"]).to_excel(file, index=False)
        st.success("Cleared!")
        st.rerun()

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("✨ AI Attendance System | Premium Dashboard UI")