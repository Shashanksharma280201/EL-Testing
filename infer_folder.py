from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import os

# Load the trained model
model = YOLO('/home/flo/Videos/EL project/codes/best.pt')

# Directory containing the images
image_folder = '/home/flo/Videos/EL project/enhanced_solar_cells_sharp'

# Define your custom output folder
output_dir = 'output'

# Run inference on all .jpg images and save results in 'output/' folder
results = model.predict(
    source=os.path.join(image_folder, '*.jpg'),
    save=True,
    conf=0.25,
    project=output_dir,  # custom folder
    name='',             # no subfolder like 'predict'; saves directly in 'output/'
    exist_ok=True        # overwrite if folder already exists
)

# Optional: Display one of the annotated results
for r in results:
    annotated_frame = r.plot()
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.title("YOLOv8 Inference Output")
    plt.axis('off')
    plt.show()
    break  # remove this break to loop through all
