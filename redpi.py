import glob
from PIL import Image

for filename in glob.glob('C:/Users/turnt/OneDrive/Desktop/Rob0Workspace/opencv-text-detection/Cropped/*.png'):
    im = Image.open(filename)
    width, height = im.size
    im = im.resize((width, 26), Image.NEAREST)
    im.save(filename, dpi=(300, 300))
