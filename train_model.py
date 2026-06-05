from deepface import DeepFace
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

dataset_path = "dataset/Faces"
X = []
y = []

print("Loading dataset...")

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):

        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(person_path, img_name)

        print("Processing:", img_path)

        try:
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            X.append(embedding)
            y.append(person)

        except:
            print("Error:", img_path)

print("\nDataset loaded!")
print("Total samples:", len(X))

if len(X) == 0:
    print("❌ No data found!")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "RandomForest": RandomForestClassifier()
}

best_model = None
best_accuracy = 0

print("\n🔍 Model Comparison:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {round(acc * 100, 2)}%")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name


print(f"\n🏆 Best Model: {best_name} with Accuracy: {round(best_accuracy*100,2)}%")

y_pred = best_model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

try:
    y_test_bin = label_binarize(y_test, classes=list(set(y)))
    y_score = best_model.predict_proba(X_test)

    fpr, tpr, _ = roc_curve(y_test_bin[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

except Exception as e:
    print("⚠️ ROC Curve skipped (multi-class issue)")

joblib.dump(best_model, "face_model.pkl")

print("\n✅ Best model saved as face_model.pkl")