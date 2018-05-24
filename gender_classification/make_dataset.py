import os
from skimage import io as io
import face_recognition as fr
import pickle
from tqdm import tqdm
from glob import glob


def get_images():
    # os.chdir("data")
    return list(glob("data/*.jpg"))


def process_images(image_list):
    everyone = []
    for image in image_list:
        if image.split('.')[1] == 'jpg':
            gender_label = image.split('_')[1]
            img = io.imread(image)
            face_embedding = fr.face_encodings(img)
            if len(face_embedding) != 1:
                continue
            person = [face_embedding[0], gender_label]
            everyone.append(person)
    return everyone


def save_as_pickle(all_data):
    with open('gender_data.pkl', 'wb') as f:
        pickle.dump(all_data, f)


def main():
    print("Gathering images from folder data...")
    images = get_images()
    if len(images) < 30:
        print('Download the full dataset from: http://mivia.unisa.it/download/2708')
        return
    print("Extracting facial features")
    data = process_images(images)
    print("Saving pickle file")
    save_as_pickle(data)


if __name__ == '__main__':
    main()
