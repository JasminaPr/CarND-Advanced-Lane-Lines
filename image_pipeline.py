import numpy as np	
import matplotlib.image as mpimg
import cv2
import pickle
import glob
from tracker import tracker

###__________camera_distortion_param_import__________
#import distortion coef. and camera matrix calculated earlier
dist_pickle = pickle.load(open("camera_cal/calibration_pickle.p", "rb")) #rb - read binary
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
	###__________Sobel threshold_x_and_y__________
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# 2) Take the derivative in x or y given orient = 'x' or 'y'
	if orient == 'x':
		sobel = cv2.Sobel(gray,cv2.CV_64F,1,0, ksize = sobel_kernel)
	elif orient == 'y':
		sobel = cv2.Sobel(gray,cv2.CV_64F,0,1, ksize = sobel_kernel)
	# 3) Take the absolute value of the derivative or gradient
	abs_sobel = np.absolute(sobel)
	# 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# 5) Create a mask of 1's where the scaled gradient magnitude 
			# is > thresh_min and < thresh_max
	sobel_binary = np.zeros_like(scaled_sobel)
	sobel_binary[(scaled_sobel>=thresh[0]) & (scaled_sobel<thresh[1])] = 1
	# 6) Return this mask as your binary_output image
	return sobel_binary
	
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
	###__________Magnitude of Sobel gradient__________
    # 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
	sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,sobel_kernel)
	sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,sobel_kernel)
    # 3) Calculate the magnitude 
	abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    #print(sobelx, abs_sobelxy)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
	scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
	sobel_binary = np.zeros_like(scaled_sobel)
	sobel_binary[(scaled_sobel>=mag_thresh[0]) & (scaled_sobel<=mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
	return sobel_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    ###__________Direction of Sobel gradient__________
    # 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
	sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = sobel_kernel)
	sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
	direction_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
	sobel_binary = np.zeros_like(direction_sobel)
	sobel_binary[(direction_sobel>=thresh[0]) & (direction_sobel<=thresh[1])] = 1
    # 6) Return this mask as your binary_output image
	return sobel_binary	
	
def color_threshold(img, s_thresh=(0, 255), r_thresh=(200,255), v_thresh=(0,255), h_thresh=(0, 100)):
    # #separate bgr channels
	# b = img[:,:,0]
	# g = img[:,:,1]
	# r_channel = img[:,:,2]
	
	# # Threshold color channel r
	# r_binary = np.zeros_like(r_channel)
	# r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
	
	#Convert to HLS color space and separate the s and h channels
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	# h_channel = hls[:,:,0]
	# l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
    # Threshold color channel s
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	# # Threshold color channel h
	# h_binary = np.zeros_like(h_channel)
	# h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
	
	#Convert to HSV color space and separate the V channel
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	# h1_channel = hsv[:,:,0]
	# s1_channel = hsv[:,:,1]
	v_channel = hsv[:,:,2]  
    # Threshold color channel v from hsv
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

	color_binary = np.zeros_like(s_channel)
	color_binary[(s_binary == 1) & (v_binary ==1)] = 1 # after trying avaliable ( (commented out) binary channels, decided for this combination
	return color_binary

def window_mask(width, height, image_ref, center, level):
	###__________Plotting prep detected line points__________
	output = np.zeros_like(image_ref)
	output[(int(image_ref.shape[0] - height*(level+1))):int((image_ref.shape[0] - height*(level))), int((center - width/2)):int((center + width/2))] = 1
	return output

###__________Pipeline of individual image process__________
#load test images
imgs = glob.glob('test_images/test*.jpg')

idx = 0
for img_name in imgs:
	idx += 1
	img = cv2.imread(img_name)
	#img = mpimg.imread(img_name)
	#undistort image
	img = cv2.undistort(img, mtx, dist, None, mtx)
	if (idx == 1):
		result = img
		saving_name = 'output_images/undistorted_' + str(idx) + '.jpg'
		cv2.imwrite(saving_name, result)
	
	##__________Creation of binary image__________
	preprocessed_image = np.zeros_like(img[:,:,0])
	
	##__________Sobel__________
	ksize = 9 # kernel size for Sobel
	gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(12, 255)) 
	grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(25, 255))
	
	## Tested, but did not show any significal improvement
	#mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
	#dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
		
	##__________Color threshold__________
	col_thresh_binary = color_threshold(img, s_thresh=(100,255), v_thresh =(50, 255))
	
	##__________Combine Sobel and Color threshold__________
	## Tested, but did not show any significal improvement
	#preprocessed_image[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (col_thresh_binary == 1)] = 255
	
	preprocessed_image[((gradx == 1) & (grady == 1)) | (col_thresh_binary == 1)] = 255
	#generate one test image for Writeup document
	if (idx == 1):
		result = preprocessed_image
		saving_name = 'output_images/binary_transform_' + str(idx) + '.jpg'
		cv2.imwrite(saving_name, result)
	
	##__________Perspective transform__________
	imshape = preprocessed_image.shape
	#define source points (ROI)
	offset_x_bottom = 0.17 #offset on lower border from left and right side in x dimension
	offset_x_up = 0.45 #offset on upper border from left and right side in x dimension #0.44
	offset_y_bottom = 0.05
	offset_y_up = 0.63 #0.61
	src = np.float32([[offset_x_up*imshape[1], imshape[0]*offset_y_up], [imshape[1]*(1-offset_x_up), imshape[0]*offset_y_up], [imshape[1]*(1-offset_x_bottom),imshape[0]*(1-offset_y_bottom)], [offset_x_bottom*imshape[1], imshape[0]*(1-offset_y_bottom)]])
	#saving one image with ROI for project Writeup
	if (idx == 1):
		src_plot = src.reshape((-1,1,2))
		img_ROI = img.copy()
		result = cv2.polylines(img_ROI,np.int32([src_plot]),True,(0,0,255))
		saving_name = 'output_images/src_ROI_result_' + str(idx) + '.jpg'
		cv2.imwrite(saving_name, result)
	
	#define destination points	
	offsetx = 0.25*imshape[1]
	offsety = 0*imshape[0]
	dst = np.float32([[offsetx,offsety], [imshape[1]-offsetx,offsety], [imshape[1]-offsetx, imshape[0]-offsety], [offsetx, imshape[0]-offsety]])
	#conduct the transform
	M = cv2.getPerspectiveTransform(src,dst)
	Minv = cv2.getPerspectiveTransform(dst,src) # inverse
	image_shape = (imshape[1], imshape[0])
	warped = cv2.warpPerspective(preprocessed_image,M,image_shape,flags=cv2.INTER_LINEAR)
	#generate one test image for Writeup document
	if (idx == 1):
		result = warped
		saving_name = 'output_images/Persp_trans_result_' + str(idx) + '.jpg'
		cv2.imwrite(saving_name, result)
	
	##__________Sliding window search__________
	window_height = 45
	window_width = 60
	margin = 30
	#based on 30 m long and 3.7 m wide lane
	ym = 30/720
	xm = 3.7/500
	#consider last n frames for average over points
	smooth_factor = 4
	
	curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = margin, My_ym = ym, My_xm = xm, Mysmooth_factor = smooth_factor)
	
	window_centroids = curve_centers.find_window_centroids(warped)
			
	##__________Sliding window visualisation__________
	l_points = np.zeros_like(warped)
	r_points = np.zeros_like(warped)
	
	leftx = []
	rightx = []
	
	for level in range(0, len(window_centroids)):
		l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
		r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
		
		#add layers of pixels found
		l_points[(l_points == 255) | (l_mask == 1)] = 255
		r_points[(r_points == 255) | (r_mask == 1)] = 255
		
		#get list of x and y points for fitting a polynomial
		leftx.append(window_centroids[level][0])
		rightx.append(window_centroids[level][1])
		
	#drawing the result of detected pixels via sliding window
	template = np.array(l_points + r_points, np.uint8)
	zero_channel = np.zeros_like(template)
	# making the points green
	template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) 
	#adding warped image to create overlapping image
	warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
	output = cv2.addWeighted(warpage, 0.8, template, 1,0.0)
	#saving one example for writeup 
	if (idx == 1):
		result = output
		saving_name = 'output_images/sliding_window_result_' + str(idx) + '.jpg'
		cv2.imwrite(saving_name, result)
	
	##__________Fitting the polynomial__________	
	ploty = np.linspace(0,warped.shape[0]-1, warped.shape[0])
	ploty_win_wise = np.arange(warped.shape[0]-window_height/2,0,-window_height)
	left_fit = np.polyfit(ploty_win_wise, leftx, 2)
	left_fitx = left_fit[0]*ploty*ploty+left_fit[1]*ploty+left_fit[2]
	left_fitx = np.array(left_fitx,np.int32)

	right_fit = np.polyfit(ploty_win_wise, rightx, 2)
	right_fitx = right_fit[0]*ploty*ploty+right_fit[1]*ploty+right_fit[2]
	right_fitx = np.array(right_fitx,np.int32)
		
	##__________Drawing the polynomial on original image__________	
	#create image to draw the lines
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	
	#recast x and y for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
	pts = np.hstack((pts_left,pts_right))
	cv2.fillPoly(color_warp,np.int_([pts]),(0,255,0))
	
	#warp back into original
	newwarp = cv2.warpPerspective(color_warp, Minv, image_shape)
	result = cv2.addWeighted(img, 1, newwarp, 0.3,0)
	
	if (idx == 1):
		saving_name = 'output_images/polynomial_result_' + str(idx) + '.jpg'
		cv2.imwrite(saving_name, result)
			
	##__________Generating final output with data__________	
	ym_per_pix = curve_centers.ym_per_pix
	xm_per_pix = curve_centers.xm_per_pix

	leftx = np.asarray(leftx)
	rightx = np.asarray(rightx)
	
	#for calculatin curvature, average from left and right lane was used.
	middle_fit = (left_fit + right_fit)/2
	middle_fit_re = np.polyfit(ploty_win_wise*ym_per_pix, np.array((leftx+rightx)/2)*xm_per_pix, 2)
	middle_curverad = (1 + (2*left_fit[0]*np.max(ploty)*ym_per_pix+middle_fit[1])**2)**(3/2)/np.absolute(2*middle_fit[0]) 
	
	##calculate offset of the car on the road
	camera_center = (left_fitx[-1] + right_fitx[-1])/2
	center_diff = (camera_center - warped.shape[1]/2)*xm_per_pix
	side_pos = 'left'
	if center_diff <=0:
		side_pos = 'right'
	
	##print the text on stream (image)
	cv2.putText(result, 'Radius of Curvature = '+str(round(middle_curverad,3))+ '(m)', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255),2)
	cv2.putText(result, 'Vehicle is = '+str(abs(round(center_diff,3)))+ 'm ' + side_pos + ' of center', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255),2)
	#saving one example for writeup 
	if (idx == 1):
		saving_name = 'output_images/final_result_' + str(idx) + '.jpg'
		cv2.imwrite(saving_name, result)
	#saving all test images in tge same folder
	saving_name = 'test_images/final_result_' + str(idx) + '.jpg'
	cv2.imwrite(saving_name, result)

