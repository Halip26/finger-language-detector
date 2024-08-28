from api import API_TOKEN
from mlforkidsimages import MLforKidsImageProject

# this one for capture photo from camera
import cv2

# to call the image file in console/terminal
import sys

# Treat this key like a password & keep it secret
key = API_TOKEN

# Train your model
myproject = MLforKidsImageProject(key)
myproject.train_model()


# Capture an image using the camera
def capture_image():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Image")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Image", frame)

        # Press 'SPACE' to capture the image
        if cv2.waitKey(1) & 0xFF == ord(" "):
            image_path = "assets/captured_image.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved to {image_path}")
            break

    cam.release()
    cv2.destroyAllWindows()
    return image_path


image_path = capture_image()

demo = myproject.prediction(image_path)

label = demo["class_name"]
confidence = demo["confidence"]

# show the result on console
print("Result: '%s' with %d%% confidence" % (label, confidence))
