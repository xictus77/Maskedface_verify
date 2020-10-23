import os
import numpy as np
import glob

from Face_fn import *
from scipy.spatial.distance import cosine

os.chdir('C:/Users/Chow Mein/PycharmProjects/')

SET_PATH = 'C:/Users/Chow Mein/PycharmProjects/MTCNN/'

known_path = SET_PATH + 'database/'
test_path = SET_PATH + 'image/'
# input test image filename here...
test_image_path = test_path + 'httpssmediacacheakpinimgcomxccccdabeaadjpg_masked2.jpg'

# Detection of faces and drawing of bounding boxes and extracting the faces using MTCNN
print('Detection of faces and drawing of bounding boxes and extracting the faces using MTCNN')
print('----------------------------------------')
print_faces(test_image_path)
draw_faces(test_image_path, faces)


print('----------------------------------------')
# The following codes will perform Face Verification With VGGFace2

# Set threshold for cosine score in compare_face function
thres_cosine = 0.6

# verify similarity by comparison and using cosine distance

def compare_face(model_scores_img1, model_scores_img2):
    for idx, face_score_1 in enumerate(model_scores_img1):
        for idy, face_score_2 in enumerate(model_scores_img2):
            score = cosine(face_score_1, face_score_2)
            if score <= thres_cosine:
                # Printing the IDs of faces and score
                print('there is a match!', idx, idy, score)
                # Displaying each matched pair of faces side by side

                plot_image = np.concatenate((img1_faces[idx], img2_faces[idy]), axis=1)
                plt.imshow(plot_image)
                plt.show()
            else:
                # Printing the IDs of faces and score
                print('this is NO match!', idx, idy, score)

                # plt.imshow(img1_faces[idx])
                # plt.show()
                # plt.imshow(img2_faces[idy])
                # plt.show()


print('Face Verification With VGGFace2')
# Performing verification with known database

img1_faces = extract_face_from_image(test_image_path)
model_scores_img1 = get_model_scores(img1_faces)

for img2_path in glob.glob(known_path + '*.*'):
    img2_faces = extract_face_from_image(img2_path)
    model_scores_img2 = get_model_scores(img2_faces)
    print('Comparing image...', img2_path)
    compare_face(model_scores_img1, model_scores_img2)
    print('Next image...')



