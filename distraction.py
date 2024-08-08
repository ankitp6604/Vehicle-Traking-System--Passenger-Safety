import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util


class VideoStream:
    
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


resolution='1280x720'
MODEL_NAME = "trained_models"
GRAPH_NAME = "model_mobinetv2_optimsize.tflite"
LABELMAP_NAME = "labels.txt"
min_conf_threshold = float(0.5)
resW, resH = resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = False


pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate


if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       


CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])


if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean =  [0.485, 0.456, 0.406] 
input_std = [0.229, 0.224, 0.225]

#map predictions to class names
def class_map(prediction):
    if(prediction==0):
        return 'safe driving'
    if(prediction==1):
        return 'texting - right'
    if(prediction==2):
        return 'talking on the phone - right'
    if(prediction==3):
        return 'texting - left'
    if(prediction==4):
        return 'talking on the phone - left'
    if(prediction==5):
        return 'operating the radio'
    if(prediction==6):
        return 'drinking'
    if(prediction==7):
        return 'reaching behind'
    if(prediction==8):
        return 'hair and makeup'
    if(prediction==9):
        return 'talking to passenger'

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)


while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

   
    if floating_model:
        input_data = input_data / 255.0
        input_data = (input_data - input_mean) / input_std
        input_data = np.float32(input_data)
  
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    tf_results = interpreter.get_tensor(output_details[0]['index'])    
    max_class = np.argmax(tf_results)
    class_name = class_map(max_class)
    
  
    cv2.putText(frame,'Predicted Class: {}'.format(str(class_name)),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

 
    cv2.imshow('Object detector', frame)


    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
videostream.stop()
