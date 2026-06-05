import cv2
import joblib
from deepface import DeepFace
import datetime
import pandas as pd
import os
model = joblib.load("face_model.pkl")
file_name = "attendance.xlsx"
if not os.path.exists(file_name):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_excel(file_name, index=False)
def mark_attendance(name):
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    df = pd.read_excel(file_name)
    if ((df["Name"] == name) & (df["Date"] == date)).any():
        return

    new_row = {"Name": name, "Date": date, "Time": time}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_excel(file_name, index=False)
cap = cv2.VideoCapture(0)

print("📷 Camera started... Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # get face embedding
        embedding = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        # predict person
        name = model.predict([embedding])[0]

        # mark attendance
        mark_attendance(name)

        # display name
        cv2.putText(frame, name, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    except Exception as e:
        pass

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()