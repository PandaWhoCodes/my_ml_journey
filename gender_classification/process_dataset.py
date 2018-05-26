from skimage import io as io
import face_recognition as fr
import pickle
from glob import glob
import cv2

def get_images(gender):
    # os.chdir("data")
    return list(glob("data/"+gender+"/*.jpg"))


def process_images(image_list,gender_label):
    everyone = []
    for image in image_list:
        print(image)
        img = cv2.imread(image)
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
    male_images = get_images("male")
    female_images = get_images("female")
    male_images = male_images[:len(female_images)]
    everyone = process_images(female_images,"female")
    everyone.extend(process_images(male_images,"male"))
    save_as_pickle(everyone)

if __name__ == '__main__':
    main()
