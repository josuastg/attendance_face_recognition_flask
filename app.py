from flask import Flask, request, jsonify
from mtcnn import MTCNN
from keras_facenet import FaceNet
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import os
from scipy import spatial
from math import radians, cos, sin, asin, sqrt
from io import BytesIO
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

load_dotenv()  # Memuat file .env

app = Flask(__name__)
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
detector = MTCNN()
embedder = FaceNet()


cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)
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

def upload_to_cloudinary(image_array, user_id, folder_name = "", absen_type = "", index = None):
    # Konversi numpy array ke file-like object (BytesIO)
    from PIL import Image
    import io

    image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    public_id = f"{user_id}_face{index + 1}" if folder_name  == "face_registration" else f"{user_id}_{absen_type}"
    result = cloudinary.uploader.upload( 
            buffer,
            public_id= public_id,
            overwrite=True,  # opsional: ganti file jika sudah ada dengan nama yang sama
            folder=folder_name,  # opsional: simpan dalam folder "faces"
            resource_type="image")
    print(result)
    return result["secure_url"]  # Link gambar

@app.route('/register-face', methods=['POST'])
def register_face():
    db = firestore.client()

    # Ambil user_id dari request (misalnya dikirim sebagai form-data)
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'user_id diperlukan'}), 400

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

    # upload photo one by one 
    photo_urls = []
    for i, face in enumerate(photos):
        try:
            url = upload_to_cloudinary(face, user_id, "face_registration", "", i);
            photo_urls.append(url)
        except Exception as e:
            print("Gagal upload ke Cloudinary:", e)

    # Simpan crop wajah
    # for i, face in enumerate(photos):
    #     img = Image.fromarray(face)
    #     img.save(f'faces/face_{len(os.listdir("faces"))}_{i}.jpg')

    # Update ke Firestore pada collection 'users'
    user_ref = db.collection('users').document(user_id)
    user_ref.set({
        'face_embedding': avg_embedding.tolist(),
        'updated_at': firestore.SERVER_TIMESTAMP,
        'photo_url': photo_urls
    }, merge=True)

    return jsonify({
        'success': True,
        'message': 'Pendaftaran wajah berhasil', # contoh salah satu angka
        'photo_url': photo_urls
    }), 200


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius bumi dalam meter
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c

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

    # Validasi input
    if not all([file, user_id, absen_type, timestamp_date, timestamp_time, longitude, latitude]):
        return jsonify({'success': False,'error': 'Semua data harus diisi'}), 400
    
    # Validasi lokasi
    lokasi_ref = db.collection('lokasi_absen').limit(1).stream()
    lokasi_doc = next(lokasi_ref, None)


    if lokasi_doc is None:
        return jsonify({'success': False, 'error': 'Lokasi kantor tidak ditemukan'}), 404

    lokasi_data = lokasi_doc.to_dict()
    user_lat = float(latitude)
    user_lon = float(longitude)
    kantor_lat = float(lokasi_data.get('latitude'))
    kantor_long = float(lokasi_data.get('longitude'))

    # Ambil properti marketing flexible
    marketing_flexible = lokasi_data.get('marketing_flexible', False)

    # Ambil data user
    user_doc = db.collection('users').document(user_id).get()
    if not user_doc.exists:
        return jsonify({'success': False, 'error': 'User tidak ditemukan'}), 404
    
    user_data = user_doc.to_dict()
    departemen = user_data.get('departement', '').lower()
    
    distance = haversine(user_lat, user_lon, kantor_lat, kantor_long)

    # Validasi lokasi
    lokasi_valid = False
    if departemen == 'marketing':
        lokasi_valid = marketing_flexible or distance <= lokasi_data.get('radius')
    else:
        lokasi_valid = distance <= lokasi_data.get('radius')

    if not lokasi_valid:
         return jsonify({'success': False, 'error': f'Lokasi absensi tidak sesuai ketentuan'}), 400

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
    .where('date', '==', timestamp_date) \
    .where('type', '==', absen_type) \
    .stream()

    # print(list(absensi_today));
    # validate agar absen masuk atau absen keluar tidak dua kali dalam sehari
    if any(absensi_today):
        validate_absen_type = 'Absen masuk' if absen_type == 'absen_masuk' else 'Absen keluar'
        return jsonify({'success': False, 'error': f'User sudah melakukan {validate_absen_type.lower()} hari ini'}), 400

    # upload photo one by one 
    photo_url = ''
    try:
        url = upload_to_cloudinary(face_array, user_id, "absen", absen_type)
        photo_url = url
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    # img = Image.fromarray(face_array)
    # img.save(f'faces/absen.jpg')
    # Simpan ke collection 'absensi'
    absen_data = {
        'user_id': user_id,
        'date': timestamp_date,
        'time': firestore.SERVER_TIMESTAMP,
        'longitude': float(longitude),
        'latitude': float(latitude),
        'type': absen_type,
        'similarity': similarity,
        'timestamp': firestore.SERVER_TIMESTAMP,
        'photo_url': photo_url
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
  app.run(host='0.0.0.0', port=5001, debug=True)
