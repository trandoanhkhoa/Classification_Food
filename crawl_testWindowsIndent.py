
"""
Created on Sat April 10:01:49 2023

@author: AnhKhoa
"""
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils import np_utils

import cv2
import numpy as np

from skimage import io
from sklearn.model_selection import train_test_split

import os
from os import listdir
from os.path import isfile, join
import datetime

# 2 fonctions for LBP
def thresholded(center, pixels):
    #print pixels
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out


def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx, idy]
    except IndexError:
        return default

#get all model
def get_all_model(folder_test):
    rs=[]
    files = os.listdir(folder_test)
    for ifile in files:
        word = ifile[:ifile.rfind(".")]
        if(word not in rs):
            rs.append(word)
    return rs
    

#get list
directory_list = list()
for root, dirs, files in os.walk("./inputphoto/", topdown=False):
    for name in dirs:
        directory_list.append(name)
#print(len(directory_list))
        
# the file of OPENCV for face detection
path = './'
folder_path_model = './backup-model/'
rs = get_all_model(folder_path_model)
for item in rs:    
    print (item)

    # load json and create model
    model_name = item
    
    json_file = open(folder_path_model+model_name +'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(folder_path_model+model_name +'.h5')
    print("Loaded model from disk")


    fileAll = open("testing_log.txt","a")
    fileDetail = open("testing_log_"+model_name+".txt","a")
    fileDetail.write (str(datetime.datetime.now())+ ";" +model_name + "\n")


    # load test photos from path
    id_folderRoot = -1
    for d in directory_list:
        print (d)
        id_folderRoot +=1
        
        # load the photos' paths firstly
        DatasetPath = []
        folder_test= './photooftest/' + d + '/'
        for i in os.listdir(folder_test)[:20]:
            DatasetPath.append(os.path.join(folder_test, i))

        imageData = []
        imageName = []

        # then read the photos, find the face in the photo
        # crop the part of face, apply LBP and resize into 46*46
        for i in DatasetPath:
            print(str(i))
            imgRead = cv2.imread(i,0) # read the cmdcm by gray
            imageName.append(str(i))
            
            cropped = imgRead
            result = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_LINEAR)  # OPENCV 3.x
            
            transformed_img = cv2.copyMakeBorder(result, 0, 0, 0, 0, cv2.BORDER_REPLICATE)

            for x in range(0, len(result)):
                for y in range(0, len(result[0])):
                    center = result[x, y]
                    #print center
                    top_left = get_pixel_else_0(result, x - 1, y - 1)
                    top_up = get_pixel_else_0(result, x, y - 1)
                    top_right = get_pixel_else_0(result, x + 1, y - 1)
                    right = get_pixel_else_0(result, x + 1, y)
                    left = get_pixel_else_0(result, x - 1, y)
                    bottom_left = get_pixel_else_0(result, x - 1, y + 1)
                    bottom_right = get_pixel_else_0(result, x + 1, y + 1)
                    bottom_down = get_pixel_else_0(result, x, y + 1)

                    values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,
                                                  bottom_down, bottom_left, left])

                    weights = [1, 2, 4, 8, 16, 32, 64, 128]
                    res = 0
                    for a in range(0, len(values)):
                        res += weights[a] * values[a]

                    transformed_img.itemset((x, y), res)

            # we only use the part (1,1) to (46,46) of the result img.
            # original img: 0-47, after resize: 1-46
            lbp = transformed_img[1:47, 1:47]  # here 1 included, 47 not included

            imageData.append(lbp)

        # apply the model to the saved face-photos
        # and print the result on the original photo
        cnt= 0
        for i in range(0,len(imageData)):
            c = np.array(imageData[i])
            c = np.array(c)
            c = c.reshape(1, 46, 46, 1)
            c = c.astype('float32')
            c /= 255

            predictions = loaded_model.predict(c)
            img = cv2.imread(imageName[i],1)
            x=10
            y=10
            h=50
            font = cv2.FONT_HERSHEY_SIMPLEX

            word = str(imageName[i])
            word = word[word.rfind("/" ) +1:word.rfind(".")]
            if predictions[0][id_folderRoot] > 0.9:
                output =word+";T;"
                cnt+=1
            else:
                output =word+";F;"
            for ii in range(0, len(directory_list)):
                output =output + (str( predictions[0][ii])+";")
            fileDetail.write(output+ "\n")
            
        fileAll.write (model_name+";"+d+";" +str(cnt)+";"+str( len(imageData))+ ";"+str( round( cnt*100/ len(imageData),6))+"\n")
    fileAll.close()
    fileDetail.close()
    print("Finish model" + item)

print ("Finish testing log")

