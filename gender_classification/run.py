import pickle
from sklearn import datasets, svm, metrics
import random
from matplotlib import pyplot as plt


def extract_data(pkl):
    with open(pkl, 'rb') as f:
        gender_dataset = pickle.load(f)
        random.shuffle(gender_dataset)
        # print(gender_dataset)
        return gender_dataset


def load_train_data(pkl):
    train_data = []
    gender_train = []

    embedding_list_test = []
    gender_label_list_test = []

    for emb, label in pkl[:len(pkl) // 2]:
        train_data.append(emb)
        gender_train.append(label)

    for emb, label in pkl[len(pkl) // 2:]:
        embedding_list_test.append(emb)
        gender_label_list_test.append(label)
    embedding_list_test.pop()
    gender_label_list_test.pop()
    print('length of embedding train list: {}'.format(len(train_data)))
    print('lenght of label train list: {}'.format(len(gender_train)))
    print('length of embedding test list: {}'.format(len(embedding_list_test)))
    print('lenght of label test list: {}'.format(len(gender_label_list_test)))
    classifier = svm.SVC(gamma='auto', kernel='rbf', C=20)
    classifier.fit(train_data, gender_label_list_test)

    expected = gender_label_list_test
    predicted = classifier.predict(embedding_list_test)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


load_train_data(extract_data("gender_data.pkl"))
