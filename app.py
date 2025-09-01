import os
from flask import Flask, request, jsonify, send_file
from deepface import DeepFace
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
GALLERY_FOLDER = 'officials'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper: Save file from request
def save_file(file_obj, folder=UPLOAD_FOLDER):
    filepath = os.path.join(folder, file_obj.filename)
    file_obj.save(filepath)
    return filepath

# API: Upload an image and search for matches in each gallery image
@app.route('/find_person', methods=['POST'])
def find_person():
    if 'query_img' not in request.files:
        return jsonify({"error": "No query image uploaded"}), 400

    query_img_file = request.files['query_img']
    query_img_path = save_file(query_img_file)

    results = []

    # Loop over each gallery image
    for gallery_img in os.listdir(GALLERY_FOLDER):
        gallery_path = os.path.join(GALLERY_FOLDER, gallery_img)
        try:
            result = DeepFace.verify(
                img1_path=query_img_path,
                img2_path=gallery_path,
                enforce_detection=False
            )
            match = result['verified']
            if match:
                # Mark face in gallery image
                img = cv2.imread(gallery_path)
                face_objs = DeepFace.extract_faces(img_path=gallery_path, enforce_detection=False)
                for face in face_objs:
                    x, y, w, h = face['facial_area'].values()
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                marked_img_path = os.path.join(UPLOAD_FOLDER, f"marked_{gallery_img}")
                cv2.imwrite(marked_img_path, img)
                results.append({
                    "gallery_image": gallery_img,
                    "match": True,
                    "marked_image_url": f"/get_image/{os.path.basename(marked_img_path)}"
                })
            else:
                results.append({
                    "gallery_image": gallery_img,
                    "match": False
                })
        except Exception as e:
            results.append({
                "gallery_image": gallery_img,
                "match": False,
                "error": str(e)
            })
    # Optionally remove uploaded query image after processing
    os.remove(query_img_path)
    return jsonify({"results": results})

# Endpoint for serving marked images
@app.route('/get_image/<filename>')
def get_image(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(file_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
