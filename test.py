import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "/home/flo/Videos/EL project/EL_Test/SH151009P636KSPC-521.jpg"  # Use the correct path
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization for contrast enhancement
equalized = cv2.equalizeHist(gray)

# Apply a denoising filter
denoised = cv2.fastNlMeansDenoising(equalized, h=10)

# Optional: Sharpening the image
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(denoised, -1, kernel)

# Display result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1), plt.imshow(gray, cmap='gray'), plt.title('Original Grayscale')
plt.subplot(1, 2, 2), plt.imshow(sharpened, cmap='gray'), plt.title('Enhanced & Sharpened')
plt.show()

# Save result if needed
cv2.imwrite("enhanced_grayscale.jpg", sharpened)
