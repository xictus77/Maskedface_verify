import os
import glob

from Face_fn import *

os.chdir('C:/Users/Chow Mein/PycharmProjects/')

SET_PATH = 'C:/Users/Chow Mein/PycharmProjects/MTCNN/'

known_path = SET_PATH + 'database/'
test_path = SET_PATH + 'image/'
# input test image filename here...
test_image_path = test_path + 'Oprah_Winfrey2_masked1.jpg'

# Detection of faces and drawing of bounding boxes and extracting the faces
print('Detection of faces and drawing of bounding boxes and extracting the faces using MTCNN')
print('----------------------------------------')
print_faces(test_image_path)
draw_faces(test_image_path, faces)


print('----------------------------------------')
# The following codes will perform Face Identification With VGGFace2

print('Face Identification With VGGFace2')
# load the photo and extract the face
print('extracting face from image...')
pixels = extract_face(test_image_path)

# convert prediction into names and display most likely results
print('predicting name from image...')
decoder(model_pred(pixels))
