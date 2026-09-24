import streamlit as st
import cv2
import numpy as np

# Import custom modules
from utils.verifier import CollegeVerifier
from utils.validator import PlateValidator
from utils.preprocess import PlatePreprocessor

# Initialize page layout
st.set_page_config(
    page_title="Dual-Layer Campus Bus ANPR System",
    page_icon="🚌",
    layout="wide"
)

st.title("🚌 Dual-Layer Campus Bus ANPR System")
st.write("Real-time License Plate Recognition & Campus Branding Verification")

# ---------------------------------------------------------
# Load OCR Engine & System Modules
# ---------------------------------------------------------
@st.cache_resource
def load_ocr():
    try:
        from paddleocr import PaddleOCR
        # Updated to use_textline_orientation to clear deprecation warning
        return PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)
    except Exception:
        try:
            from paddleocr import PaddleOCR
            return PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
        except Exception as e:
            st.error(f"Failed to load PaddleOCR engine: {e}")
            return None

ocr_engine = load_ocr()
verifier = CollegeVerifier()
validator = PlateValidator()
preprocessor = PlatePreprocessor()

def detect_and_crop_plate(image):
    """
    Plate Crop Extractor:
    Crops lower region where plates sit.
    """
    h, w, _ = image.shape
    plate_crop = image[int(h * 0.65):int(h * 0.98), int(w * 0.15):int(w * 0.85)]
    return plate_crop

# ---------------------------------------------------------
# Streamlit File Upload Pipeline
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Bus Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    full_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB), width="stretch")

    with col2:
        st.subheader("Decision Engine Analysis")

        # -----------------------------------------------------
        # STEP 1: License Plate Extraction & Preprocessing
        # -----------------------------------------------------
        raw_plate_crop = detect_and_crop_plate(full_image)
        processed_plate_crop = preprocessor.process(raw_plate_crop)

        # Run PaddleOCR on Processed Plate Crop
        detected_text = "NO_TEXT"
        confidence = 0.0

        if ocr_engine is not None and processed_plate_crop is not None:
            try:
                ocr_result = ocr_engine.ocr(processed_plate_crop)
                if ocr_result and ocr_result[0]:
                    lines = []
                    confidences = []
                    for line in ocr_result[0]:
                        if line and len(line) > 1 and len(line[1]) > 0:
                            lines.append(str(line[1][0]))
                            confidences.append(float(line[1][1]))

                    raw_ocr_str = "".join(lines).strip()
                    confidence = float(np.mean(confidences)) * 100 if confidences else 0.0
                    
                    # Safe validation check preventing 'index out of range'
                    if raw_ocr_str:
                        _, detected_text = validator.validate(raw_ocr_str)
                    else:
                        detected_text = "NO_TEXT"
            except Exception as e:
                st.warning(f"Plate OCR processing note: {e}")

        # Display Plate Results
        st.image(cv2.cvtColor(processed_plate_crop, cv2.COLOR_BGR2RGB), caption="Plate Crop", width=180)
        st.markdown("**Plate Number**")
        st.write(f"### `{detected_text}`")
        st.markdown("**Plate OCR Confidence**")
        st.write(f"### `{confidence:.2f}%`")

        st.divider()

        # -----------------------------------------------------
        # STEP 2: Layer 2 Verification (Branding + DB Match)
        # -----------------------------------------------------
        banner_passed, banner_msg = verifier.verify_bus_branding(
            ocr_engine=ocr_engine,
            full_image=full_image,
            detected_plate=detected_text
        )

        # -----------------------------------------------------
        # STEP 3: Final Decision Engine Output
        # -----------------------------------------------------
        if banner_passed:
            st.success(f"**Layer 2 (Banner Detection):** PASSED — {banner_msg}")
            st.success("### Final Decision: **AUTHORIZED (CAMPUS BUS)**")
        else:
            st.error(f"**Layer 2 (Banner Detection):** FAILED — {banner_msg}")
            st.warning("### Final Decision: **UNAUTHORIZED (NO VALID CAMPUS BRAND/REGISTRATION)**")