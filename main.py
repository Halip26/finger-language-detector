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

image_path = sys.argv[1]

demo = myproject.prediction(image_path)

label = demo["class_name"]
confidence = demo["confidence"]

# show the result on console
print("Result: '%s' with %d%% confidence" % (label, confidence))
