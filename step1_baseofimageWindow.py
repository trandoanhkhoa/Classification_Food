import os
import sys
from os import listdir
from os.path import isfile, join

import cv2
import numpy
import numpy as np
from numpy import *
from random import randint

'''
Function support clone data
Edit by: Anh Khoa
Date: April 07,2023
'''
def larger(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(0, 24)

    pts1 = np.float32([[num, num], [cols - num, num], [num, rows - num], [cols - num, rows - num]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst

def smaller(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(0, 24)

    pts1 = np.float32([[num, num], [cols - num, num], [num, rows - num], [cols - num, rows - num]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts2, pts1)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst

def lighter(img):
    # copy the basic picture, avoid the change of the basic one
    dst = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    # get the number of rows and cols of picture
    rows, cols = dst.shape[:2]
    # take a random number to use
    num = randint(20,50)

    for xi in range(0, cols):
        for xj in range(0, rows):
            for i in range(0, 3):
                if dst[xj, xi, i] <= 255 - num:
                    dst[xj, xi, i] = int(dst[xj, xi, i] + num)
                else:
                    dst[xj, xi, i] = 255
    return dst

def darker(img):
    # copy the basic picture, avoid the change of the basic one
    dst = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(20,50)

    for xi in range(0, cols):
        for xj in range(0, rows):
            for i in range(0, 3):
                if dst[xj, xi, i] >= num:
                    dst[xj, xi, i] = int(dst[xj, xi, i] - num)
                else:
                    dst[xj, xi, i] = 0
    return dst

def moveright(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,num],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def moveleft(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,-num],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def movetop(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,0],[0,1,-num]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def movebot(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,0],[0,1,num]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def turnright(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(3,6)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -num, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def turnleft(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(3,6)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), num, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def changeandsave(name,time,choice,img,i):
    # the new name of changed picture is "changed?.png",? means it's the ?-th picture changed
    name = './changedphoto/' + str(name) + '/' +str(time) + '_changed' + str(i) + '.jpg'
    # do different changes by the choice
    if choice == 1:
        newimg = larger(img)
    elif choice == 2:
        newimg = smaller(img)
    elif choice == 3:
        newimg = lighter(img)
    elif choice == 4:
        newimg = darker(img)
    elif choice == 5:
        newimg = moveright(img)
    elif choice == 6:
        newimg = moveleft(img)
    elif choice == 7:
        newimg = movetop(img)
    elif choice == 8:
        newimg = movebot(img)
    elif choice == 9:
        newimg = turnleft(img)
    elif choice == 10:
        newimg = turnright(img)
    # save the new picture
    cv2.imwrite(name, newimg)


'''
Main code clone data x100 images (LBP)
'''
#get list
directory_list = list()
for root, dirs, files in os.walk("./inputphoto/", topdown=False):
    for name in dirs:
        directory_list.append(name)


print("step 1 clone")
newpath = "./changedphoto/"

#create new folder changedphoto
for d in directory_list:
    if not os.path.exists(newpath+ d):
        os.makedirs(newpath+d)

# take photos in folder, change each photo into 100 photos
for d in directory_list: 
    for root, dirs, files in os.walk("./inputphoto/"+ d + "/", topdown=False):
        for filenameimg  in files: #full code
        #for filenameimg  in files[:200]: #demo code
            print (d+"/"+filenameimg)
            img = cv2.imread('./inputphoto/'+ d +'/'+ filenameimg,1)
            
            for i in range(1,3):   #demo code clone 2 image from input  
                # take a random number as the choice
                choice = randint(1,10)
                changeandsave(d,filenameimg,choice,img,i)

# and we need to use opencv to apply face detection and LPB of every image
print("step 2 LBP")
path = './'
# 2 fonctions for LBP
def thresholded(center, pixels):
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

# we apply LBP and face detection to all photos and save the new photos
indexfolder = 0
for d in directory_list:
    indexfolder += 1
    mypath = './changedphoto/'+ d
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    images = numpy.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(join(mypath, onlyfiles[n]))
        # transform the new image into GRAY
        newgray = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)
        cropped = newgray
        # here cropped is still in size of w*h, we need an image in 48*48, so change it
        result = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_LINEAR)  # OPENCV 3.x

        # then use LBP to this image
        # copy result as transformed_img
        transformed_img = cv2.copyMakeBorder(result, 0, 0, 0, 0, cv2.BORDER_REPLICATE)

        for x in range(0, len(result)):
            for y in range(0, len(result[0])):
                center = result[x, y]
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

        # save the final image
        name = './CNNdata/'+str(indexfolder)+'_'+d+'_' + str(n) + '.jpg' 
        cv2.imwrite(name, lbp)
