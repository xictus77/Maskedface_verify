import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams
from mtcnn.mtcnn import MTCNN
from PIL import Image
import glob
from numpy import asarray
import tensorflow as tf

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine



os.chdir('C:/Users/Chow Mein/PycharmProjects/')

SET_PATH = 'C:/Users/Chow Mein/PycharmProjects/MTCNN/'

known_path = SET_PATH + 'database/'
test_path = SET_PATH + 'image/'
test_image_path = test_path + 'httpssmediacacheakpinimgcomxccccdabeaadjpg_masked2.jpg'

faces = []
detector = MTCNN()


def extract_face_from_image(image_path, required_size=(224, 224)):
    # load image and detect faces
    image = Image.open(image_path)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    image = asarray(image)

    # detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face_boundary = image[y1:y2, x1:x2]

        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

    return face_images


def highlight_faces(image_path, faces):
    # display image
    image = plt.imread(image_path)
    plt.imshow(image)

    ax = plt.gca()

    # for each face, draw a rectangle based on coordinates
    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x, y), width, height,
                                fill=False, color='red')
        ax.add_patch(face_border)
    plt.show()

# draw each face separately
def draw_faces(image_path, faces):
    # load the image
    data = plt.imread(image_path)
    faces = detector.detect_faces(data)
    # plot each face as a subplot
    for i in range(len(faces)):
    # get coordinates
        x1, y1, width, height = faces[i]['box']
        x2, y2 = x1 + width, y1 + height
        # define subplot
        plt.subplot(1, len(faces), i+1)
        plt.axis('off')
    # plot face
        plt.imshow(data[y1:y2, x1:x2])
    # show the plot
    plt.show()


def print_faces(image_path):
    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)
    for face in faces:
        print(face)
        print("Number of faces detected: ", len(faces))
    highlight_faces(image_path, faces)


def get_model_scores(faces):
    samples = asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    model = VGGFace(model='resnet50',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg')

    # perform prediction
    return model.predict(samples)


# Set threshold for cosine score
thres_cosine = 0.6


# verify similarity by comparison and using cosine distance

def compare_face(model_scores_img1, model_scores_img2):
    for idx, face_score_1 in enumerate(model_scores_img1):
        for idy, face_score_2 in enumerate(model_scores_img2):
            score = cosine(face_score_1, face_score_2)
            if score <= thres_cosine:
                # Printing the IDs of faces and score
                print('this is a match', idx, idy, score)
                # Displaying each matched pair of faces side by side

                plot_image = np.concatenate((img1_faces[idx], img2_faces[idy]), axis=1)
                plt.imshow(plot_image)
                plt.show()

                # plt.imshow(img1_faces[idx])
                # plt.show()
                # plt.imshow(img2_faces[idy])
                # plt.show()


# Detection of faces and drawing of bounding boxes and extracting the faces

print_faces(test_image_path)
draw_faces(test_image_path, faces)


# Perform Face Verification With VGGFace2

img1_faces = extract_face_from_image(test_image_path)

model_scores_img1 = get_model_scores(img1_faces)
path_name = known_path

for img2_path in glob.glob(path_name + '*.*'):
    img2_faces = extract_face_from_image(img2_path)
    model_scores_img2 = get_model_scores(img2_faces)
    print('Comparing image...', img2_path)
    compare_face(model_scores_img1, model_scores_img2)
    print('Next image...')