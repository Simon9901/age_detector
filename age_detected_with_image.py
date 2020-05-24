# import the necessary packages
import numpy as np
import  os
import cv2
import argparse

# GPU support
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# construct the argument parse and the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image.')
ap.add_argument('-f', '--face', required=True,
                help='path to face detector model directory.')
ap.add_argument('-a', '--age', required=True,
                help='path to age detectir model directory.')
ap.add_argument('-c', '--confidence', type=float, default=0.5,
                help='minimum probability to filter weak detections.')

args = vars(ap.parse_args())

# define the list of age buckets our age detector will predict
AGE_BUCKETS = ['(0-2)', '(4-6', '(8-12)', '(15-20)', '(25-32)', '(38-43)',
               '(48-53)', '(60-100)']

# load our serialized face detector model from disk
print('[INFO] loading the face detector model...')
prototxtPath = os.path.sep.join([args['face'], 'deploy.prototxt'])
weightsPath = os.path.sep.join([args['face'], 'res10_300x300_ssd_iter_140000.caffemodel'])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our serilaized age detector model from disk
print('[INFO] loading the age detector model...')
prototxtPath = os.path.sep.join([args['age'], 'age_deploy.prototxt'])
weightsPath = os.path.sep.join([args['age'], 'age_net.caffemodel'])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the input image and construct an input blob for the image
image = cv2.imread(args['image'])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detection
print('[INFO] computing face detections...')
faceNet.setInput(blob)
detections = faceNet.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e. probability) associated with the prediction
    confidence = detections[0,0,i,2]

    # filter out weak detections by ensuring the confidence is greater than the ,imi,u, confidence
    if confidence > args['confidence']:
        # compute the (x,y)-coordinates of the bounding box for the object
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (x_start, y_start, x_end, y_end) = box.astype('int')

        # extract the ROI of the face and then construct a blob from 'only' the face ROI
        face = image[y_start:y_end, x_start:x_end]
        faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227,227),
                                         (78.4263377603, 87.7689143744, 114.895847746),
                                         swapRB=False)

        # make predictions on the age and find the age bucket with the largest corresponding
        # probability
        ageNet.setInput(faceBlob)
        preds = ageNet.forward()
        i = preds[0].argmax()
        age = AGE_BUCKETS[i]
        ageConfidence = preds[0][i]

        # display the predicted age to our terminal
        text = '{}:{:.2f}%'.format(age, ageConfidence*100)
        print('[INFO] {}'.format(text))

        # draw the bounding box of the face along with the associated predicted age
        if y_start -10 > 10:
            y = y_start - 10
        else:
            y = y_start + 10

        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0,0,255), 2)
        cv2.putText(image, text, (x_start, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0,0,255), 2)

# display the outptu image
cv2.imshow('Image', image)
cv2.waitKey(0)