import cv2
import os
dataset_path = "dataset/Faces"
processed_path = "processed_dataset"

if not os.path.exists(processed_path):
    os.makedirs(processed_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    save_path = os.path.join(processed_path, person)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            face = cv2.resize(img, (160, 160))
            save_file = os.path.join(save_path, img_name)
            cv2.imwrite(save_file, face)
        else:
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))

                save_file = os.path.join(save_path, img_name)
                cv2.imwrite(save_file, face)

print("Preprocessing Done ✅")