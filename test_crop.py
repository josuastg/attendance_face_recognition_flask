from flask import Flask, request, jsonify, send_file
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
import os
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

detector = MTCNN()

def detect_and_crop(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, "Invalid image"

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)

    if not results:
        return None, "No face detected"

    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)
    face = rgb[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    return face, None

@app.route('/test-crop', methods=['POST'])
def test_crop():
    file = request.files.get('photo')
    if not file:
        return jsonify({'error': 'No photo provided'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    face, err = detect_and_crop(filepath)
    if err:
        return jsonify({'error': err}), 400

    # Simpan hasil crop ke JPEG di memory
    img = Image.fromarray(face)
    buf = BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)

    return send_file(buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
