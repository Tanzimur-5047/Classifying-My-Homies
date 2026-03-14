import cv2
import numpy as np
from mtcnn import MTCNN
import os

detector = MTCNN()

def crop_largest_face(image_folder):

    for filename in os.listdir(image_folder):

        img_path = os.path.join(image_folder, filename)

        img = cv2.imread(img_path)

        if img is not None and img.shape[0] == 224 and img.shape[1] == 224:
            print(f"Already processed: {filename} — skipping")
            continue

        if img is None:
            print(f"Skipping non-image file: {filename}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            faces = detector.detect_faces(img_rgb)
        except Exception as e:
            print(f"MTCNN failed on {filename}: {e} — skipping")
            continue

        if len(faces) == 0:
            print(f"No face found in: {filename} — skipping")
            continue


        largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])



        x, y, w, h = largest_face['box']
        if w < 20 or h < 20:
            print(f"Face too small in: {filename} — skipping")
            continue




        margin = int(0.1 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)

        face_crop = img[y1:y2, x1:x2]


        face_resized = cv2.resize(face_crop, (224, 224))

        jpg_path = os.path.splitext(img_path)[0] + ".jpg"
        cv2.imwrite(jpg_path, face_resized)


        if jpg_path != img_path:
            os.remove(img_path)

        print(f"Processed: {filename}")


if __name__ == "__main__":
    folders = [
        "data/train/Aninda",
        "data/train/Himel",
        "data/train/Sukomal",
        "data/train/Unknown"
        ]

    for folder in folders:
        print(f"\n--- Processing {folder} ---")
        crop_largest_face(folder)

    print("\nDone! All faces cropped and saved.")