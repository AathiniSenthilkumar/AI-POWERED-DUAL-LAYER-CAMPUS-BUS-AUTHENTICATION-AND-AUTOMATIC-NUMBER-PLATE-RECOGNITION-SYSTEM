import os
import cv2

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True,
    enable_mkldnn=False
)

image = cv2.imread("Output/plate_0.jpg")

if image is None:
    print("Error: Could not read Output/plate_0.jpg")
else:
    results = ocr.predict(image)

    print("\n" + "=" * 45)
    print("PARSED ANPR OUTPUT:")
    print("=" * 45)
    
    for res in results:
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])
        for text, score in zip(texts, scores):
            print(f"Detected Plate : {text}")
            print(f"Confidence     : {score * 100:.2f}%")
            
    print("=" * 45)