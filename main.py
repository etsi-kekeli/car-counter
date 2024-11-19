import cv2 as cv
import tensorflow as tf
import time
import tensorflow_hub as hub
import numpy as np
import utils

hub_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
model = hub.load(handle=hub_handle)

detector = model.signatures['default']

stream = cv.VideoCapture('data/autoroute.mp4')
windowName = "Autoroute"
cv.namedWindow(windowName, cv.WINDOW_NORMAL)

while cv.waitKey(1) != 27:
    hasFrame, frame = stream.read()

    if not hasFrame:
        break

    H, W, _ = frame.shape

    tensor = tf.image.convert_image_dtype(frame, tf.float32)[tf.newaxis, ...]

    start_time = time.time()
    result = detector(tensor)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    boxes = np.array(result["detection_boxes"])
    boxes[:, [0, 2]] *= H
    boxes[:, [1, 3]] *= W

    n_frame = utils.highlightCars(
        frame, np.array(result["detection_scores"]), boxes.astype(int), np.array(result["detection_class_entities"]), .08)

    cv.imshow(windowName, n_frame)


stream.release()
cv.destroyWindow(windowName)
