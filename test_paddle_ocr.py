from paddleocr import PaddleOCR
import cv2
import os

print("Initializing PaddleOCR...")
# Initialize PaddleOCR - first time will download models
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

# Test on a simple image first
test_img_path = "C:\Users\JaydenBrown\plates\car3.jpg"  # Update with your test image

# Find a test image
for file in os.listdir(test_img_path):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        test_img = os.path.join(test_img_path, file)
        print(f"Testing on: {test_img}")
        
        img = cv2.imread(test_img)
        if img is not None:
            # Run OCR
            result = ocr.ocr(img, cls=True)
            
            print("\nOCR Results:")
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    print(f"Text: '{text}', Confidence: {confidence:.2f}")
            break

print("\nPaddleOCR test complete!")
