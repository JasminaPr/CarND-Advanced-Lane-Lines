import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

imgs = glob.glob('calibration*.jpg')

#point lists
imgpoints = []
objpoints = []

# object points preparation
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
#print(objp)

idx = 0
for image in imgs:
	idx += 1
    #read image and convert image to grayscale
	img = cv2.imread(image)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray, cmap = 'gray')
	
    #find corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    #print('ret', ret, 'corners', corners)
	
    #append to list to connect corners with indexes of corners, if ret = true
	if ret == True:
		#print('working on: ',  image)
		imgpoints.append(corners)
		objpoints.append(objp)
		#print(objpoints,imgpoints)
		img = cv2.drawChessboardCorners(img, (9,6) , corners, ret)
		#saves images of detected corners
		write_name = 'corners_detected' + str(idx) + '.jpg' #it saves different ordering
		cv2.imwrite(write_name,img)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

##Create undistorted example for documentation
img = cv2.imread('../test_images/test1.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx) # undistorted images
write_name = '../output_images/undistorted/unidstorted_test1.jpg' 
cv2.imwrite(write_name,dst)

#saving the values in a list, will be loaded afterwards.
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("./calibration_pickle.p", "wb")) #open for writing in binary mode