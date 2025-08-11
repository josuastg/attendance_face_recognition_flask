# Attendance Face Recognition API (Flask + Firebase)

🚀 Sistem absensi berbasis **pengenalan wajah** yang dibangun menggunakan **Flask** dan **Firebase**.  
Aplikasi ini menyediakan API untuk mendaftarkan wajah, melakukan absensi (masuk/keluar), manajemen pengguna, hingga integrasi penyimpanan foto ke **Cloudinary**.

---

## 📌 Fitur Utama
- **Pendaftaran Wajah (Face Registration)** menggunakan MTCNN + FaceNet
- **Absensi Otomatis** berdasarkan kecocokan wajah
- **Validasi Lokasi** menggunakan perhitungan *Haversine*
- **Manajemen Pengguna** (tambah, hapus, bulk create)
- **Integrasi Firebase** (Firestore & Authentication)
- **Upload Foto ke Cloudinary**
- **Ekspor Excel** untuk data pengguna yang gagal dibuat saat proses bulk

---

## 🛠️ Teknologi yang Digunakan
- **Python** (Flask)
- **MTCNN** untuk deteksi wajah
- **FaceNet** untuk ekstraksi *face embedding*
- **Firebase Admin SDK** (Firestore + Auth)
- **Cloudinary** untuk penyimpanan foto
- **OpenPyXL** untuk ekspor Excel
- **dotenv** untuk konfigurasi environment

---

## 🚀 Instalasi & Menjalankan
1. **Clone Repository**
- git clone https://github.com/josuastg/attendance_face_recognition_flask.git
- cd attendance_face_recognition_flask
2. **Buat Virtual Environment & Install Dependencies**
- python -m venv venv
- source venv/bin/activate  # MacOS/Linux
- venv\Scripts\activate     # Windows
- pip install -r requirements.txt

3. **Siapkan File Konfigurasi** serviceAccountKey.json → kredensial Firebase dan.env → konfigurasi Cloudinary & variabel lain*
- CLOUDINARY_CLOUD_NAME=your_cloud_name
- CLOUDINARY_API_KEY=your_api_key
- CLOUDINARY_API_SECRET=your_api_secret

4. **Jalankan Aplikasi** (Firestore + Auth)
- python app.py
- Akses API di http://127.0.0.1:5000

---

## 📂 Struktur Proyek
```bash
attendance_face_recognition_flask/
│
├── app.py                 # Main API application
├── basic_embedding.py     # Script utilitas embedding wajah
├── test_crop.py           # Script testing crop wajah
├── requirements.txt       # Daftar dependencies Python
├── .gitignore             # File ignore Git
└── serviceAccountKey.json # Credential Firebase (tidak disertakan di repo publik)
