import cv2
import glob
import numpy as np

images = glob.glob("images/**/*.*", recursive=True)
# getting the image moments
FILE_NAME = "Hus_moments.txt"
final_list = []
for image in images:
    image1 = cv2.imread(image)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    final_list.append(list(cv2.HuMoments(cv2.moments(image1)).flatten()))

to_write = ""
for items in final_list:
    for item in items:
        to_write += str(item) + "\t"
    to_write += "\n"

# print(final_list)
with open(FILE_NAME,"w") as f:
    f.write(to_write)