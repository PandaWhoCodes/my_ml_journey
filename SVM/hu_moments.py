import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)

images = glob.glob("images/**/*.*", recursive=True)
# getting the image moments
FILE_NAME = "Hus_moments.txt"
final_list = []
all_moments = np.empty((0,7))

for image in images:
    image1 = cv2.imread(image)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(image1, 1)
    hu_moments = cv2.HuMoments(moments)
    final_list.append(list(hu_moments))
    all_moments = np.append(all_moments, hu_moments, 0)

to_write = ""
for items in final_list:
    for item in items:
        to_write += str(item) + "\t"
    to_write += "\n"

# print(final_list)
with open(FILE_NAME,"w") as f:
    f.write(to_write)
