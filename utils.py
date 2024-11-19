import cv2 as cv
import numpy as np


objectsOfInterest = ['Car', 'Vehicule']


def highlightCars(frame, scores, boxes, classnames, threshold=0.1):
    isVehicule = np.logical_or(classnames == b'Car', classnames == b'Vehicule')
    mask = np.logical_and(isVehicule, scores >= threshold)

    filteredBoxes = boxes[mask]

    for box in filteredBoxes:
        cv.rectangle(frame, pt1=(box[1], box[0]), pt2=(box[3], box[2]),
                     color=(255, 0, 255), thickness=5, lineType=cv.LINE_8)

    return frame
