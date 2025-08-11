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
