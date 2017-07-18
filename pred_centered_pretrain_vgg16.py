import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os
import sys
from keras.models import load_model
from vis.visualization import visualize_saliency

# test image paths
sample_submission = pd.read_csv("../input/sample_submission.csv")
img_path = "../input/test/"

test_names = []
file_paths = []

for i in range(len(sample_submission)):
    test_names.append(sample_submission.iloc[i, 0])
    file_paths.append(img_path + str(int(sample_submission.iloc[i, 0])) + '.jpg')

test_names = np.array(test_names)


# image resize & centering & crop
def centering_image(img):
    size = [256, 256]

    img_size = img.shape[:2]

    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized


X_test = []
for i, file_path in enumerate(file_paths):
    sys.stdout.write("\r {0} from total {1} images".format(file_path, len(file_paths)))
    sys.stdout.flush()

    # read image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ratio_list = [0.0, 0.1, 0.2, 0.3]

    sub_images = []
    for r in ratio_list:
        l, w, _ = img.shape
        dl, dw = int(r * l), int(r * w)
        sub_img = img[dl:(l - dl), dw:(w - dw), :]

        # resize
        if (sub_img.shape[0] > sub_img.shape[1]):
            tile_size = (int(sub_img.shape[1] * 256 / sub_img.shape[0]), 256)
        else:
            tile_size = (256, int(sub_img.shape[0] * 256 / sub_img.shape[1]))

        # centering
        sub_img = centering_image(cv2.resize(sub_img, dsize=tile_size))

        # output 224*224px
        sub_img = sub_img[16:240, 16:240]
        sub_img = sub_img.astype('float32')
        sub_img /= 255.0
        sub_images.append(sub_img)
    X_test.append(sub_images)
X_test = np.array(X_test)

# remember shape before reshaping
inshape = X_test.shape
X_test = X_test.reshape((-1, ) + inshape[-3:])

# load pretrained model
model = load_model('VGG16-transferlearning.model')

# make predictions
y_test_pred = model.predict(X_test)
y_test_pred = y_test_pred.reshape(inshape[:2])
# sorted in ascending order
y_test_pred.sort(axis=1)

y_test_pred_mean = np.mean(y_test_pred, axis=1)
y_test_pred_bimax = np.mean(y_test_pred[:, -2:], axis=1)

# write predictions to file
def submission_to_csv(test_preds, file_path):
    sample_submission = pd.read_csv("../input/sample_submission.csv")
    for i, name in enumerate(test_names):
        sample_submission.loc[sample_submission['name'] == name, 'invasive'] = test_preds[i]
    sample_submission.to_csv(file_path, index=False)

submission_to_csv(test_preds=y_test_pred_mean, file_path= 'submit_mean.csv')
submission_to_csv(test_preds=y_test_pred_bimax, file_path= 'submit_bimax.csv')
