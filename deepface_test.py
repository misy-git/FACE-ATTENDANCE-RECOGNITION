from deepface import DeepFace

result = DeepFace.verify(
    img1_path="processed_dataset/Akshay_kumar/0.jpg",
    img2_path="processed_dataset/Akshay_kumar/1.jpg"
)

print(result)