from turtle import Turtle
import numpy as np
import cv2
import time

path_model = "./models/"

# Read Network
model_name = "model-f6b98070.onnx" # MiDaS v2.1 Large
# model_name = "model-small.onnx" # MiDaS v2.1 Small

# Load the DNN model
model = cv2.dnn.readNetFromONNX(path_model + model_name)

if (model.empty()):
    print("Could not load model. Check path")

model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Webcam
cap = cv2.VideoCapture(0)


while cap.isOpened():
    #Read the image
    success, img = cap.read()
    
    img_height, img_width, channels = img.shape
    
    #Start time to calculate FPS
    start = time.time()
    
    
    # Create Blob from Input Image
    # MiDaS v2.1 large (Scale : 1/255, Size : 384x384, Mean Subtraction : (123.675, 116.28, 103.53), Channels Order: RGB)
    blob = cv2.dnn.blobFromImage(1/255., (384, 384), (123.675, 116.28, 103.53), True)
    
    # MiDaS v2.1 small (Scale : 1/255, Size : 256x256, Mean Subtraction : (123.675, 116.28, 103.53), Channels Order: RGB)
    # blob = cv2.dnn.blobFromImage(1/255., (256, 256), (123.675, 116.28, 103.53), True)
    
    
    # Set input to the model
    model.setInput(blob)
    
    
    # Make froward pass in model
    output = model.forward()
    
    output = output[0, :, :]
    output = cv2.resize(output, (img_width, img_height))
    
    
    # Normalize the output
    output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX) # NOrmalize values to be between 0 and 1
    
    
    # End time
    
    end = time.time()
    fps = 1 - (end-start)
    
    # Show FPS
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX)
    
    cv2.imshow("image", img)
    cv2.imshow("Depth map", output)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()