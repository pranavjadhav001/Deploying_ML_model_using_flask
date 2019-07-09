import cv2
import os
#import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import load_model
import dlib
import tensorflow as tf

model = load_model('blood.hdf5')
model._make_predict_function()
def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 123.68
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 103.939
        # 'RGB'->'BGR'
        #x = x[:, :, :, ::-1]
    return x

def facer(image):
    cropped_blood = []    
    face_detector = dlib.get_frontal_face_detector()
    detected_faces = face_detector(image,1)
    if len(detected_faces) > 0:
        for i,face_rect in enumerate(detected_faces):
            coordinates= [face_rect.left(),face_rect.bottom(),face_rect.right(),face_rect.top()]
            coordinates = [0 if j < 0 else j for j in coordinates]
                #img1 = cv2.rectangle(img,(coordinates[0],coordinates[1]),(coordinates[2],coordinates[3]),(255,0,0),2)
            cropped_blood.append(image[coordinates[3]:coordinates[1],coordinates[0]:coordinates[2]])
        return cropped_blood
    else:
        return None
    
                
        
    
def prediction(image_path,model_path='blood.hdf5'):
    image = cv2.imread(image_path)
    faces = facer(image)
    all_face = []
    scores = []
    global model
    if faces != None:
        for i in faces:
            img = cv2.resize(i,(224,224))
            all_face.append(img)
            img = np.expand_dims(img,axis=0)
            img = preprocess_input(img.astype('float64'))
            ans = model.predict(img)
            word = 'blood' if np.argmax(ans) == 1 else 'no_blood'
            scores.append(word)
        return all_face,scores
    else:
        return "No face found"
    
