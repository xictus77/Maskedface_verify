
from numpy import expand_dims
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray


from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from keras_vggface.vggface import VGGFace

# Initialize variables

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

    # create a vggface model object (2 models to choose from resnet50 and senet50)
    model = VGGFace(model='resnet50',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg')
    # summarize input and output shape
    # print('Inputs: %s' % model.inputs)
    # print('Outputs: %s' % model.outputs)

    # perform prediction
    return model.predict(samples)

def model_pred(faces):
    samples = faces.astype('float32')
    samples = expand_dims(samples, axis=0)

    # prepare the data for the model
    samples = preprocess_input(samples, version=2) # change the version to 1 if model used is vgg16

    # create a vggface model object
    model = VGGFace(model='senet50')
                    # options of two model='resnet50' and 'senet50'
                    # if vgg16 is used, change the version to 1
    # summarize input and output shape
    print('Inputs: %s' % model.inputs)
    print('Outputs: %s' % model.outputs)

    # perform prediction
    yhat = model.predict(samples)
    return yhat

# convert prediction into names
def decoder(yhat):
    results = decode_predictions(yhat)
# display most likely results
    for result in results[0]:
        print('The likely candidate is %s with confidence level of %.3f%%' % (result[0], result[1]*100))

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = plt.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array