from paddleocr import PaddleOCR
import cv2

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Test on an image
img_path = '../data/uploads/your_test_image.jpg'  # Update this
img = cv2.imread(img_path)

# Run OCR
result = ocr.ocr(img, cls=True)

# Print results
for line in result:
    if line:
        for word_info in line:
            text = word_info[1][0]
            confidence = word_info[1][1]
            print(f"Text: {text}, Confidence: {confidence}")
