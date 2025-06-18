from flask import Flask, request, jsonify
from mtcnn import MTCNN
from keras_facenet import FaceNet
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
detector = MTCNN()
embedder = FaceNet()


def extract_face(file):
    img = Image.open(file).convert('RGB')
    img_array = np.asarray(img)

    # results akan berisi list wajah yang ditemukan.
    # Ambil bounding_box dari wajah pertama:
    results = detector.detect_faces(img_array)

    if not results:
        return None

    # Ini menghasilkan crop wajah dari gambar asli.
    x, y, width, height = results[0]['box']
    x, y = abs(x), abs(y)
    face = img_array[y:y + height, x:x + width]

# FaceNet memerlukan input dengan resolusi 160x160 piksel. jadi perlu diresize 160 x 160
    face_image = Image.fromarray(face).resize((160, 160))
    return np.asarray(face_image)


def register_face():
    photos = []
    for i in range(3):
        # Terima Gambar (request)
        # Gambar diterima dalam bentuk file.
        # Dibaca dan dikonversi ke NumPy array menggunakan PIL + OpenCV.
        file = request.files.get(f'photo{i}')
        if file:
            face_array = extract_face(file)
            if face_array is not None:
                photos.append(face_array)
            else:
                return jsonify({"status": "error",'error': f'Wajah tidak terdeteksi di photo{i}'}), 400

    if len(photos) != 3:
        return jsonify({'error': '3 wajah valid diperlukan'}), 400

    # Hitung rata-rata embedding
    embeddings = embedder.embeddings(photos)
    avg_embedding = np.mean(embeddings, axis=0)

    # Simpan embedding ke file (atau nanti ke Firebase)
    np.save(f'embeddings/embedding_{len(os.listdir("embeddings"))}.npy', avg_embedding)

    # Simpan foto asli
    for i, face in enumerate(photos):
        img = Image.fromarray(face)
        img.save(f'faces/face_{len(os.listdir("faces"))}_{i}.jpg')

    return jsonify({'message': 'Pendaftaran wajah berhasil', 'embedding': avg_embedding[10].tolist()}), 200