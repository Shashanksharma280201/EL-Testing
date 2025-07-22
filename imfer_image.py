from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# Load the trained model
model = YOLO('/home/flo/Videos/EL project/codes/best.pt')  # <- change this to your model path

# Path to the image you want to test
image_path = '/home/flo/Videos/EL project/enhanced_ultra_clean.jpg'  # <- change this to your image

# Run inference
results = model.predict(source=image_path, save=True, conf=0.25)

# Show results
annotated_frame = results[0].plot()  # Annotated image (NumPy array)
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.title("YOLOv8 Inference Output")
plt.axis('off')
plt.show()
