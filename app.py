import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

# Page config
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="✋",
    layout="wide"
)

# Title
st.title("✋ Real-Time Sign Language Detection")
st.markdown("Detects A-Z letters and 0-9 numbers using ensemble CNN models")

# ===========================
# LOAD MODELS (with caching) saved_models_effb0
# ===========================
@st.cache_resource
def load_models():
    model_dir = Path("saved_models_effb0")
    
    model_paths = [
        model_dir / "final_model_sign_language.h5",
    ]
    
    models = []
    for path in model_paths:
        if path.exists():
            try:
                model = tf.keras.models.load_model(str(path))
                models.append(model)
                st.sidebar.success(f"✓ Loaded {path.name}")
            except Exception as e:
                st.sidebar.error(f"✗ Error loading {path.name}: {e}")
        else:
            st.sidebar.error(f"✗ Not found: {path.name}")
    
    return models

with st.spinner("Loading models..."):
    models = load_models()

if len(models) == 0:
    st.error("❌ No models found! Please check your model paths.")
    st.stop()

st.success(f"✓ Loaded {len(models)} ensemble models")

# ===========================
# CLASS LABELS
# ===========================
num_classes = models[0].output_shape[1]
if num_classes == 36:
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                    'U', 'V', 'W', 'X', 'Y', 'Z']
elif num_classes == 26:
    class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                    'U', 'V', 'W', 'X', 'Y', 'Z']
else:
    class_labels = [str(i) for i in range(num_classes)]

# ===========================
# MEDIAPIPE SETUP
# ===========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ===========================
# PREDICTION FUNCTIONS
# ===========================
def ensemble_predict(models, roi, class_labels):
    if roi is None or roi.size == 0:
        return None, None, None
    
    try:
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        
        preds = np.zeros(models[0].output_shape[1])
        for model in models:
            preds += model.predict(img, verbose=0)[0]
        preds /= len(models)
        
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])
        label = class_labels[class_id] if class_id < len(class_labels) else str(class_id)
        
        return label, class_id, confidence
    except Exception as e:
        return None, None, None

def extract_hand_roi(frame, hand_landmarks, margin=30):
    h, w, _ = frame.shape
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    
    x_min = max(0, int(min(x_coords)) - margin)
    x_max = min(w, int(max(x_coords)) + margin)
    y_min = max(0, int(min(y_coords)) - margin)
    y_max = min(h, int(max(y_coords)) + margin)
    
    roi = frame[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max, y_max)

def process_image(image, models, class_labels):
    """Process uploaded image"""
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                roi, (x_min, y_min, x_max, y_max) = extract_hand_roi(frame, hand_landmarks)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                label, class_id, confidence = ensemble_predict(models, roi, class_labels)
                
                if label is not None:
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), label, confidence
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, None

# ===========================
# STREAMLIT UI
# ===========================
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Select Mode:", ["📷 Upload Image", "🎥 Live Webcam"])

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

# ===========================
# MODE 1: UPLOAD IMAGE
# ===========================
if mode == "📷 Upload Image":
    st.header("📷 Upload an Image")
    uploaded_file = st.file_uploader("Choose a hand sign image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Prediction Result")
            with st.spinner("Analyzing..."):
                result_img, label, confidence = process_image(image, models, class_labels)
                
                if label is not None:
                    st.image(result_img, use_container_width=True)
                    
                    if confidence >= confidence_threshold:
                        st.success(f"### **Predicted Sign: {label}**")
                        st.metric("Confidence", f"{confidence:.2%}")
                        st.progress(confidence)
                    else:
                        st.warning(f"**Low Confidence: {label}**")
                        st.metric("Confidence", f"{confidence:.2%}")
                else:
                    st.error("❌ No hand detected")

# ===========================
# MODE 2: LIVE WEBCAM
# ===========================
elif mode == "🎥 Live Webcam":
    st.header("🎥 Live Webcam Detection")
    
    run_webcam = st.checkbox("Start Webcam", value=False)
    
    FRAME_WINDOW = st.image([])
    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:
            
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                label_text = "No hand detected"
                conf_value = 0.0
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )
                        
                        # Extract ROI
                        roi, (x_min, y_min, x_max, y_max) = extract_hand_roi(frame, hand_landmarks)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        
                        # Predict
                        label, class_id, confidence = ensemble_predict(models, roi, class_labels)
                        
                        if label is not None and confidence > confidence_threshold:
                            # Draw prediction on frame
                            text = f"{label}: {confidence:.2f}"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                            
                            # Background for text
                            cv2.rectangle(frame,
                                        (x_min, y_min - text_size[1] - 15),
                                        (x_min + text_size[0], y_min - 5),
                                        (0, 255, 0), -1)
                            
                            cv2.putText(frame, text, (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            
                            label_text = f"**{label}**"
                            conf_value = confidence
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Display prediction below
                if conf_value > confidence_threshold:
                    prediction_placeholder.success(f"Detected: {label_text}")
                    confidence_placeholder.progress(conf_value)
                else:
                    prediction_placeholder.info(label_text)
                    confidence_placeholder.empty()
        
        cap.release()
        st.info("Webcam stopped")

# ===========================
# FOOTER
# ===========================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info(f"""
**Ensemble Models:** {len(models)}  
**Total Classes:** {len(class_labels)}  
**Supported Signs:**  
{', '.join(class_labels)}
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Use good lighting
- Plain background works best
- Hold hand steady and clear
- Face palm towards camera
- Adjust confidence threshold in settings
""")
