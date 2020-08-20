from PIL import Image
import os

path = 'faces/lfw-deepfunneled/'

destination = 'faces/training/notrong/'

i = 0
for folder in os.listdir(path):
    for fname in os.listdir(path + folder):
        i += 1
        img = Image.open(path + folder + '/' + fname)
        copy = img.copy()
        copy.save(destination + fname)