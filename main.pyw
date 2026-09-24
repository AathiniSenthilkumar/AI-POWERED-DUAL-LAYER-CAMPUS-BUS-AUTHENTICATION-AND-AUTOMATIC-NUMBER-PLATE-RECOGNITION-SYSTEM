import os

# Set environment flags at the very top
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import cv2
from utils.detector import PlateDetector
from utils.preprocess import PlatePreprocessor
from utils.paddleocr_engine import PaddleEngine
from utils.validator import PlateValidator
from utils.saver import ResultSaver

def run_pipeline(image_path="Images/bus.jpg", model_path="Models/best.pt"):
    print("=" * 50)
    print("INITIALIZING CAMPUS BUS ANPR PIPELINE")
    print("=" * 50)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image from {image_path}")
        return

    # Initialize components
    detector = PlateDetector(model_path)
    preprocessor = PlatePreprocessor()
    ocr_engine = PaddleEngine()
    validator = PlateValidator()
    saver = ResultSaver(output_dir="Output")

    print("\n[STEP 1] Running YOLO License Plate Detection...")
    detected_plates = detector.detect(image)

    if not detected_plates:
        print("[WARNING] No license plates detected.")
        return

    print(f"[SUCCESS] Found {len(detected_plates)} plate(s). Processing...")

    processed_results = []

    for idx, plate_info in enumerate(detected_plates):
        crop = plate_info["crop"]
        bbox = plate_info["bbox"]

        # Preprocess
        enhanced_crop = preprocessor.process(crop)

        # OCR Reading
        raw_text, confidence = ocr_engine.read(enhanced_crop)

        # Validation
        is_valid, final_plate = validator.validate(raw_text)

        conf_pct = confidence * 100

        print("\n" + "-" * 40)
        print(f"PLATE #{idx + 1} RESULTS:")
        print("-" * 40)
        print(f"BBox Location : {bbox}")
        print(f"Raw OCR Text  : {raw_text}")
        print(f"Confidence    : {conf_pct:.2f}%")
        print(f"MoRTH Format  : {'VALID' if is_valid else 'INVALID'}")
        print(f"Final Plate   : {final_plate}")
        print("-" * 40)

        # Append to results collection for saving
        processed_results.append({
            "crop": crop,
            "enhanced": enhanced_crop,
            "bbox": bbox,
            "final_plate": final_plate,
            "confidence": conf_pct,
            "is_valid": is_valid
        })

    # Save all output artifacts automatically
    saver.save_all(image, processed_results)

if __name__ == "__main__":
    run_pipeline(image_path="Images/bus.jpg", model_path="Models/best.pt")