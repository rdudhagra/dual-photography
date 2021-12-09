import cv2
import numpy as np
import skimage.io
from block import clear_blocks, floodlight_blocks

from constants import CAMERA_H, CAMERA_W, CAMERA_ID

cap = None
vid = None

dark_image = np.zeros((CAMERA_H, CAMERA_W))


def camera_init():
    global cap
    global vid
    global dark_image

    vid = cv2.VideoWriter(
        "capture_recording.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (CAMERA_W, CAMERA_H),
    )

    cap = cv2.VideoCapture(CAMERA_ID)
    clear_blocks()
    dark_image = camera_read_gray()
    floodlight_blocks()


def camera_read():
    global cap
    global vid

    if cap is None:
        raise Exception("Camera not initialized")

    _, frame = cap.read()

    # Resize image
    frame = cv2.resize(frame, (CAMERA_W, CAMERA_H))

    # Write frame to video
    vid.write(frame)

    # skimage.io.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # skimage.io.show()

    return frame


def camera_read_gray():
    """
    Reads a frame from the camera and converts it to grayscale

    Returns:
        np.ndarray: Grayscale image, normalized to [0, 1]
    """
    global dark_image

    gray_img = cv2.cvtColor(camera_read(), cv2.COLOR_BGR2GRAY)
    # Normalize the image
    if gray_img.max() > 0:
        gray_img = gray_img / gray_img.max()

    # skimage.io.imshow(gray_img)
    # skimage.io.show()

    return gray_img - dark_image


def camera_close():
    global cap
    global vid

    cap.release()
    vid.release()

