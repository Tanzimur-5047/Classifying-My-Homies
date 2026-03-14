import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN
from PIL import Image
import os
import base64

MODEL_PATH = 'model/best_model_ft.keras'
CLASS_NAMES = ['Aninda', 'Himel', 'Sukomal', 'Unknown']
CONFIDENCE_THRESHOLD = 0.70
IMG_SIZE = 224
MEDIA_DIR = 'app/media'

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_detector():
    return MTCNN()

model = load_model()
detector = load_detector()

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_normalized = face_resized / 255.0
    return np.expand_dims(face_normalized, axis=0)

def predict_face(face_img):
    input_tensor = preprocess_face(face_img)
    predictions = model.predict(input_tensor, verbose=0)
    confidence = float(np.max(predictions))
    class_idx = int(np.argmax(predictions))
    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence
    return CLASS_NAMES[class_idx], confidence

def draw_box(image, x, y, w, h, name, confidence):
    color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
    label = f"{name} ({confidence:.0%})"
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x, y - 25), (x + label_size[0], y), color, -1)
    cv2.putText(image, label, (x, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image

def play_media(name):
    video_path = os.path.join(MEDIA_DIR, f"{name}.mp4")
    audio_path = os.path.join(MEDIA_DIR, f"{name}.mp3")

    if os.path.exists(video_path):
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode()
        st.markdown(f'''
            <video width="100%" autoplay controls>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
        ''', unsafe_allow_html=True)
    elif os.path.exists(audio_path):
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        st.markdown(f'''
            <audio autoplay controls>
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
        ''', unsafe_allow_html=True)
    else:
        st.info(f"🎬 Media placeholder for {name} — add {name}.mp4 or {name}.mp3 to app/media/")

st.set_page_config(
    page_title="Friend Classifier",
    page_icon="👥",
    layout="centered"
)

st.title("👥 Friend Classifier")
st.markdown("Upload a photo and I'll tell you if Aninda, Himel or Sukomal is in it!")
st.divider()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image_pil, caption="Uploaded Image", use_column_width=True)
    st.divider()

    with st.spinner("Detecting faces..."):
        try:
            faces = detector.detect_faces(image_np)
        except Exception:
            faces = []

    if len(faces) == 0:
        st.warning("😕 No faces detected in this image.")
    else:
        results = []

        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)

            if w < 20 or h < 20:
                continue

            margin = int(0.1 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image_bgr.shape[1], x + w + margin)
            y2 = min(image_bgr.shape[0], y + h + margin)

            face_crop = image_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            name, confidence = predict_face(face_rgb)

            if name is not None:
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'box': (x, y, w, h)
                })

        recognized = [r for r in results if r['name'] is not None]

        if len(recognized) == 1:
            person = recognized[0]
            if person['name'] == 'Unknown':
                st.warning("### 🚫 Unknown person detected!")
                st.metric("Confidence", f"{person['confidence']:.1%}")
                st.divider()
                play_media('Unknown')
            else:
                st.success(f"### 👤 {person['name']} detected!")
                st.metric("Confidence", f"{person['confidence']:.1%}")
                st.divider()
                st.markdown(f"**Playing media for {person['name']}:**")
                play_media(person['name'])

        elif len(recognized) > 1:
            annotated = image_bgr.copy()
            for r in recognized:
                x, y, w, h = r['box']
                annotated = draw_box(annotated, x, y, w, h, r['name'], r['confidence'])

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detected Faces", use_column_width=True)

            st.markdown("### 👥 Results:")
            for r in recognized:
                if r['name'] == 'Unknown':
                    st.warning(f"**Unknown person** — {r['confidence']:.1%} confidence")
                else:
                    st.success(f"**{r['name']}** — {r['confidence']:.1%} confidence")

        else:
            st.warning("😕 Faces detected but none recognized.")
            st.caption("Confidence was below the threshold for all detected faces.")