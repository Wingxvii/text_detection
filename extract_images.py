import cv2
import time

input = 'C:/Users/turnt/OneDrive/Desktop/Rob0Workspace/ctr.mp4'
output = "C:/Users/turnt/OneDrive/Desktop/Rob0Workspace/Outputs/{}-{:04}.{}"

ext = "jpg"
width = 244
height = 244
interp = cv2.INTER_NEAREST
frequency = 5
volume = 4000
basename = "Test"
crop = True

video = cv2.VideoCapture(input)

counter = 0
iterator = 0
skip = video.get(cv2.CAP_PROP_FPS) / 5

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
        cv2.imwrite(output.format(basename, counter, ext), frame)
        iterator = 0

    else:
        break

video.release()

cv2.destroyAllWindows()
