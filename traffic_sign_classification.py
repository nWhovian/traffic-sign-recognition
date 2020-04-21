import matplotlib as matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import random
import csv
import cv2
import time
from skimage import transform
from skimage.util import random_noise
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy import ndimage
import matplotlib.pyplot as plt


def readTrafficSigns(rootpath):
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 42 classes
    for c in range(0, 43):
        print(c)
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()
    return images, labels


def readTestData(rootpath):
    test_images = []  # images
    test_labels = []  # corresponding labels

    gtFile = open(rootpath + '/' + 'GT-final_test' + '.csv')  # annotations file
    gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
    next(gtReader)  # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        test_images.append(plt.imread(rootpath + '/' + row[0]))  # the 1th column is the filename
        test_labels.append(row[7])  # the 8th column is the label
    gtFile.close()
    return test_images, test_labels


def reshape_resize(size, images):
    for i in range(len(images)):
        # add padding borders to the shortest side
        img_shape = images[i].shape
        if img_shape[0] > img_shape[1]:
            delta = img_shape[0] - img_shape[1]
            images[i] = cv2.copyMakeBorder(images[i], 0, 0, delta // 2, delta - delta // 2, cv2.BORDER_REPLICATE)
        if img_shape[1] > img_shape[0]:
            delta = img_shape[1] - img_shape[0]
            images[i] = cv2.copyMakeBorder(images[i], delta // 2, delta - delta // 2, 0, 0, cv2.BORDER_REPLICATE)

        # resize image
        images[i] = cv2.resize(images[i], (size, size))
    return images


def cross_validation(images, labels):
    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []

    dictionary = {}  # where key=class(label), value = all images in that class

    for i in range(len(labels)):
        label = labels[i]
        if label in dictionary:
            dictionary.get(label).append(images[i])
        else:
            dictionary[label] = [images[i]]

    for label in dictionary:
        size = len(dictionary[label])
        for i in range(size):
            if i / size <= 0.8:
                train_images.append(dictionary[label][i])
                train_labels.append(label)
            else:
                validation_images.append(dictionary[label][i])
                validation_labels.append(label)

    return train_images, train_labels, validation_images, validation_labels


# show diagram
def show_class_dist(labels):
    y_pos = list(range(43))
    performance = [labels.count(str(y)) for y in y_pos]

    plt.bar(y_pos, performance, align='center', alpha=0.8)
    plt.ylabel('Examples')
    plt.title('43 classes frequencies (training set)')

    plt.show()


# randomly transform image
def aug(image):
    # rotate image
    if random.choices([1, 0], weights=[0.7, 0.3])[0] == 1:
        image = transform.rotate(image, random.uniform(-45, 45))

    # add noise to the image
    if random.choices([1, 0], weights=[0.05, 0.95])[0] == 1:
        image = random_noise(image, mode="s&p", clip=True)

    # blur image
    n = random.choices([0, 1, 2, 3], weights=[0.6, 0.2, 0.1, 0.1])[0]
    image = ndimage.uniform_filter(image, size=(n, 1, 1))

    # rescale intensity image
    if random.choices([1, 0], weights=[0.1, 0.9])[0] == 1:
        v_min, v_max = np.percentile(image, (10, 50))
        image = exposure.rescale_intensity(image, in_range=(v_min, v_max))

    return image


def augmentation(train_images, train_labels):
    dictionary = {}

    for i in range(len(train_labels)):
        if train_labels[i] in dictionary:
            dictionary[train_labels[i]].append(train_images[i])

        else:
            dictionary[train_labels[i]] = [train_images[i]]

    y_pos = list(range(43))
    performance = [train_labels.count(str(y)) for y in y_pos]
    n = np.amax(performance)
    for i in range(len(performance)):
        if performance[i] < n:
            length = len(dictionary[str(i)])
            for _ in range(n - performance[i]):
                j = random.randrange(0, length - 1)
                img = aug(dictionary[str(i)][j])
                train_images.append(img)
                train_labels.append(str(i))

    return train_images, train_labels


# normalize image (divide all values by 255)
def normalize(images):
    images = np.array(images)
    images = images / 255

    new_set = []

    for i in range(len(images)):
        train = images[i]
        new_set.append(np.ravel(train))

    return new_set


def shuffle(a, b):
    combined = list(zip(a, b))
    random.shuffle(combined)

    a[:], b[:] = zip(*combined)

    return a, b


def main():
    images, labels = readTrafficSigns("Images")
    test_images, test_labels = readTestData("Images_test")

    size = int(input("specify the size of the preprocessed images:"))
    print("press 1 for augmentation, 0 - otherwise")
    f = input()
    is_augmentation = False
    if f == '1': is_augmentation = True

    # reshape and resize the images
    images = reshape_resize(size, images)
    test_images = reshape_resize(size, test_images)

    print("reshaping & resizing completed")

    # divide data to train set and validation set
    train_images, train_labels, validation_images, validation_labels = cross_validation(images, labels)

    print("dividing to train/validate completed")

    # perform augmentation
    if is_augmentation == True:
        train_images, train_labels = augmentation(train_images, train_labels)
        print("augmentation completed")

    # normalize train_set & validation_set and shape to vector
    train_set = normalize(train_images)
    validation_set = normalize(validation_images)
    test_set = normalize(test_images)

    print("normalization completed")

    # shuffle all
    train_set, train_labels = shuffle(train_set, train_labels)
    validation_set, validation_labels = shuffle(validation_set, validation_labels)

    print("begin learning...")

    begin_time = time.time()
    clf = RandomForestClassifier(random_state=1, n_estimators=100, n_jobs=-1)
    clf.fit(train_set, train_labels)
    end_time = time.time()

    print("finish learning, time = ", str(end_time - begin_time))

    predicted_val = clf.predict(validation_set)
    val_accuracy = accuracy_score(validation_labels, predicted_val)
    print("validation score = " + str(val_accuracy))

    predicted_test = clf.predict(test_set)
    test_accuracy = accuracy_score(test_labels, predicted_test)
    print("test score = " + str(test_accuracy))


if __name__ == "__main__":
    main()