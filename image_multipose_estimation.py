import cv2
from imread_from_url import imread_from_url

from HRNET import HRNET, PersonDetector
from HRNET.utils import ModelType, filter_person_detections

# Initialize Pose Estimation model
model_path = "models/hrnet_coco_w48_384x288.onnx"
model_type = ModelType.COCO
hrnet = HRNET(model_path, model_type, conf_thres=0.5)

# Initialize Person Detection model
person_detector_path = "models/yolov5s6.onnx"
person_detector = PersonDetector(person_detector_path)

# Read image (using URL as fallback)
img_url = "https://wavewear.cc/cdn/shop/articles/parkour_1.jpg?v=1691646892&width=1100"
local_image_path = "input.png"

# Try loading the local image first
img = cv2.imread(local_image_path)
if img is None:
    print(f"Local image '{local_image_path}' not found. Attempting to load from URL.")
    img = imread_from_url(img_url)
    if img is None:
        raise FileNotFoundError(f"Could not load image from URL: {img_url}")

# Detect People in the image
detections = person_detector(img)
ret, person_detections = filter_person_detections(detections)

if ret:
    # Estimate the pose in the image
    total_heatmap, peaks = hrnet(img, person_detections)

    # Draw Model Output
    img = hrnet.draw_pose(img)

    # Draw detections (optional)
    # img = person_detector.draw_detections(img)

# Display and save the output image
cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
cv2.imshow("Model Output", img)
cv2.imwrite("doc/img/output.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
