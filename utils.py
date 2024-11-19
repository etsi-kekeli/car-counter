import cv2 as cv
import numpy as np


def highlightCars(frame, scores, boxes, classnames, threshold=0.1):
    isVehicule = np.logical_or(classnames == b'Car', classnames == b'Vehicule')
    mask = np.logical_and(isVehicule, scores >= threshold)

    filteredBoxes = boxes[mask]

    for box in filteredBoxes:
        cv.rectangle(frame, pt1=(box[1], box[0]), pt2=(box[3], box[2]),
                     color=(255, 0, 255), thickness=2, lineType=cv.LINE_8)

    text = f"{len(filteredBoxes)} vehicles detected"
    fontScale = 2
    fontFace = cv.FONT_HERSHEY_PLAIN
    fontColor = (0, 0, 0)
    fontThickness = 2

    cv.putText(frame, text, (150, 30), fontFace, fontScale,
               fontColor, fontThickness, cv.LINE_AA)

    return frame
