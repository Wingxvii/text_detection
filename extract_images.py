"""
.. module:: extract_images
   :synopsis: Extracts images from a video

.. moduleauthor:: John Wang <john.wang@uoit.net>
"""

import cv2
import time

def extract(input, basename, crop=False, volume=5000, frequency=5, height=244, width=244, ext="jpg", output="C:/Users/turnt/OneDrive/Desktop/Rob0Workspace/Outputs/"):
    """
    Extracts images from a video, then saves it to a file

    :param input: filepath for input video
    :type input: str
    :param basename: name for output images
    :type basename: str
    :param crop: wether to crop or resize
    :type crop: bool
    :param volume: max amount of images to output
    :type volume: int
    :param frequency: amount of images to skip
    :type frequency: int
    :param height: height size for image output size
    :type height: int
    :param width: width size for image output size
    :type width: int
    :param ext: file extention
    :type ext: str
    :param output: output file path
    :type output: str
    """

    form = "{}-{:04}.{}"
    interp = cv2.INTER_NEAREST
    video = cv2.VideoCapture(input)

    counter = 0
    iterator = 0
    skip = video.get(cv2.CAP_PROP_FPS) / frequency

    while(video.isOpened()):
        ret, frame = video.read()

        # Skip Frames
        if iterator <= skip:
            iterator += 1
            continue

        if ret is True and counter < volume:
            counter += 1
            # Do operation
            if crop is True:
                frame = frame[0:360, 140:500]
            frame = cv2.resize(frame, (width, height), interpolation=interp)
            # Save
            cv2.imwrite(output + form.format(basename, counter, ext), frame)
            iterator = 0

        else:
            break

    video.release()
    cv2.destroyAllWindows()
