from flask import Flask, request, jsonify
from models.face_detector import detector
from models.embedder import embedder
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
# import os
from scipy import spatial


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'API Flask di Vercel jalan'})

# pastikan folder simpan tersedia
# os.makedirs("faces", exist_ok=True)
# os.makedirs("embeddings", exist_ok=True)

# MTCNN
# Gambar utuh → image
# Deteksi wajah → dapat koordinat (x, y, w, h)
# Crop wajah → ambil bagian image[y:y+h, x:x+w]
# Hasil crop ini kemudian:
# di-resize ke 160x160
# diubah menjadi float
# dikirim ke FaceNet untuk menghasilkan embedding
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



@app.route('/register-face', methods=['POST'])
def register_face():
    from firebase_admin import firestore
    db = firestore.client()

    # Ambil user_id dari request (misalnya dikirim sebagai form-data)
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id diperlukan'}), 400

    photos = []
    for i in range(3):
        file = request.files.get(f'photo{i}')
        if file:
            face_array = extract_face(file)
            if face_array is not None:
                photos.append(face_array)
            else:
                return jsonify({'success': False ,'error': f'Wajah tidak terdeteksi di photo{i}'}), 400

    if len(photos) != 3:
        return jsonify({'success': False, 'error': '3 wajah valid diperlukan'}), 400

    # Embedding + rata-rata
    embeddings = embedder.embeddings(photos)
    avg_embedding = np.mean(embeddings, axis=0)

    # Simpan embedding ke file lokal (jika masih ingin disimpan lokal)
    # np.save(f'embeddings/embedding_{len(os.listdir("embeddings"))}.npy', avg_embedding)

    # Simpan crop wajah
    # for i, face in enumerate(photos):
    #     img = Image.fromarray(face)
    #     img.save(f'faces/face_{len(os.listdir("faces"))}_{i}.jpg')

    # Update ke Firestore pada collection 'users'
    user_ref = db.collection('users').document(user_id)
    user_ref.set({
        'face_embedding': avg_embedding.tolist(),
        'face_registered': True,
        'updated_at': firestore.SERVER_TIMESTAMP
    }, merge=True)

    return jsonify({
        'success': True,
        'message': 'Pendaftaran wajah berhasil', # contoh salah satu angka
    }), 200

@app.route('/absen', methods=['POST'])
def absen():
    # Ambil data dari request
    file = request.files.get('photo')
    user_id = request.form.get('user_id')
    absen_type = request.form.get('type')
    timestamp_date = request.form.get('date')
    timestamp_time = request.form.get('time')
    longitude = request.form.get('longitude')
    latitude = request.form.get('latitude')
    location_name = request.form.get('location_name')

    # Validasi input
    if not all([file, user_id, absen_type, timestamp_date, timestamp_time, longitude, latitude, location_name]):
        return jsonify({'success': False,'error': 'Semua data harus diisi'}), 400

    # Ekstrak wajah dari gambar
    face_array = extract_face(file)
    if face_array is None:
        return jsonify({'success': False, 'error': 'Wajah tidak terdeteksi'}), 400

    # Hitung embedding wajah dari foto
    new_embedding = embedder.embeddings([face_array])[0]

    # Ambil data embedding dari Firestore (collection users)
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()

    if not user_doc.exists:
        return jsonify({'success': False, 'error': 'User tidak ditemukan'}), 404

    user_data = user_doc.to_dict()
    stored_embedding = np.array(user_data.get('face_embedding'))

    if stored_embedding is None:
        return jsonify({'success': False, 'error': 'User belum memiliki data wajah'}), 404

    # Hitung jarak cosine
    similarity = 1 - spatial.distance.cosine(stored_embedding, new_embedding)

    # Threshold minimal untuk validasi wajah
    if similarity < 0.7:
        return jsonify({'success': False, 'error': 'Wajah tidak cocok'}), 401

    absensi_today = db.collection('absensi') \
    .where('user_id', '==', user_id) \
    .where('time', '==', timestamp_date) \
    .where('type', '==', absen_type) \
    .get()

    # validate agar absen masuk atau absen keluar tidak dua kali dalam sehari
    if absensi_today:
        validate_absen_type = 'Absen masuk' if absen_type == 'absen_masuk' else 'Absen keluar'
        return jsonify({'success': False, 'error': f'User sudah melakukan {validate_absen_type.lower()} hari ini'}), 409
    
    # Simpan ke collection 'absensi'
    absen_data = {
        'user_id': user_id,
        'date': timestamp_date,
        'time': timestamp_time,
        'longitude': float(longitude),
        'latitude': float(latitude),
        'location_name': location_name,
        'type': absen_type,
        'similarity': similarity,
        'timestamp': firestore.SERVER_TIMESTAMP
    }

    absen_ref = db.collection('absensi').add(absen_data)[1]  # get doc ID
    absen_id = absen_ref.id

    update_absen_ref = db.collection('absensi').document(absen_id)
    update_absen_ref.set({
        'id': absen_id
    }, merge=True)
    
    validate_absen_type = 'Absen masuk berhasil' if absen_type == 'absen_masuk' else 'Absen keluar berhasil'

    return jsonify({
        'success': True,
        'message': validate_absen_type,
        'absen_id': absen_id,
        'similarity': similarity
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
