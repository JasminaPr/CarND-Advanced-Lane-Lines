**Advanced Lane Finding Project**

This project was done following examples of code provided during lectures Camera Calibration, Gradients and Color Spaces and Advanced Computer Vision. Also for creation of this project instructions provided in video "Self-Driving Car Project Q&A | Advanced Lane Finding" were used. (https://www.youtube.com/watch?v=vWY8YUayf9Q&feature=youtu.be). 

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[Chessboard_points_image]: ./camera_cal/corners_detected2.jpg "Chessboard_points_image"
[Undistorted_Chessboard]: ./output_images/undistorted_Chessboard_image_1.jpg "Undistorted_Chessboard"
[Undistorted]: ./output_images/undistorted_1.jpg "Undistorted"
[binary_transform] : ./output_images/binary_transform_1.jpg "binary_transform"
[src_ROI]: ./output_images/src_ROI_result_1.jpg "src_ROI" 
[Persp_trans]: ./output_images/Persp_trans_result_1.jpg "Persp_trans"
[sliding_window]: ./output_images/sliding_window_result_1.jpg "sliding_window"
[polynomial_result]: ./output_images/polynomial_result_1.jpg "polynomial_result"
[final_result]: ./output_images/final_result_1.jpg "final_result"
[output_video]: ./project_video_output.mp4 "output_video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Writeup / README

### Camera Calibration

Code for Camera Calibration step is located in "\camera_cal\cam_cal.py".

"cam_cal.py" has to be called individually to calculate camera matrix (mtx) and distortion coefficients (dist), which will be used later to undistort camera images (or videos) containing lanes. Pipeline of "cam_cal.py" is following:
	- initialize lists of two point sets: "imgpoints" and "objpoints". "imgpoints" will contain pixel coordinates (2D) of each chessboard corner detected. "objpoints" will contain matching 3D real world coordinates of detected chessboard corner.
	- create "objp" - list of indices of points, from [0,0,0] until [8,5,0] for 9x6 corner grid
	- get a list of image names containing chessboard pictures in the folder "\camera_cal\".
	- loop trough each image. Convert into gray and find chessboard corners using cv2.findChessboardCorners().
	- if corners are found, append 2D and 3D coordinates to "imgpoints" and "objpoints" respectively. 
	- Draw points on image for visualisation using cv2.drawChessboardCorners(). Example of the chessboard image with detected corners: [Chessboard_points_image]. Rest of the images can be found in the same folder (./camera_cal/corners_detected*.jpg).
	- after looping through images is done, calculate camera matrix (mtx) and distortion coefficients (dist) using "cv2.calibrateCamera()" based on detected points.
	- save the values into dist_pickle dictionary. 
	- example of undistorted Chessboard image is given in [Undistorted_Chessboard].

### Pipeline (single images)

Pipeline for single images will follow "video_gen.py" (process_image()). Additionally, "image_pipeline.py" is provided for running the pipeline on test images provided by the project and generating image examples given in this description. Pipeline in "video_gen.py" and "image_pipeline.py" is the same.

#### 1. Provide an example of a distortion-corrected image.

Pipeline starts with calling cv2.undistort() function ("video_gen.py", line 123). Input for the function is distorted image and mtx and dist (camera matrix and distortion coefficients) saved as result from "\camera_cal\cam_cal.py". Example of undistorted image is provided in [Undistorted].

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

To generate binary image couple of methods were used: Sobel - related methods and color thresholding.

Sobel operator is being used to calculate gradients in both x and y direction (gradx and grady). Function is defined as abs_sobel_thresh() ("video_gen.py", line 17). Input to the function is image, direction of sobel operator, kernel size, pixel threshold. This function converts the image to grayscale, then calls cv2.Sobel() function. All pixels within the threshold range are activated in the binary image (sobel_binary).

Additionaly, two more functions are defined: magnitude of Sobel gradient (mag_thresh(), "video_gen.py", line 37) and direction of Sobel gradient (dir_threshold(), "video_gen.py", line 55). Magnitude of Sobel gradient is square root of the squares of the individual x and y gradients. Given result is then being thresholded and placed into binary output (mag_binary). Direction of the gradient is inverse tangent (arctangent) of the y gradient divided by the x gradient (dir_binary). Results are then thresholded.

As result of testing Sobel related methods, it was observed that introducting magnitude and gradient (((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))) did not bring significant improvement. Therefore overall binary  Sobel filter image is produced as result of: (grady & gradx), which is then combined with color threshold explained as follows.  

Color gradient is defined in color_threshold() function given in "video_gen.py", line 73. Function takes as input image and desired threshold ranges for individual color channels. Following channels were experimented with: r from bgr, s and h from hls and v channel from hsv. Based on results, it was decided to use s_channel threshold from hls and r_channel threshold from rgb ((s_binary == 1)&(r_binary == 1), "video_gen.py", line 105.

Color threshold was added to create final binary image ("video_gen.py", line 144). Example of the image is given in [binary_transform].

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transform is done in "video_gen.py", lines 151-176. To conduct perspective fransform, two sets of rectangle points were given: source and destination. Source points (src) are of similar shape as given in first project of this nanodegree. They are defined in line 158. Visualisation of those points can be seen in [src_ROI]. Destination points are defined as a rectangle where source points should be mapped. Points are defined as "dst" in line 170. 

Once destination and source points are defined, cv2.warpPerspective() is used to get M, matrix needed to transofm undistorted image into bird view perspective. Later vice versa will have to be done, so Minv is also calculated. Example of perspective transform can be seen in image [Persp_trans]. Lines appear parallel in the given image.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane line pixels are detected using sliding window method ("video_gen.py", lines 182-194). In first part, class tracker ("tracker.py") is being called. Class contains find_window_centroids() method. In this method, line starting points are found by summing and convolving first quarter of the image. After this is found, further points are found by similar method - summing lyers defined by window_height parameters along x axis and convolving with a window defined by window_width parameter. After points for given frame are found, to avoid sudden jumps between frames averaging should be peformed in tracker class. 

(Note to reviewer: I was following code given in "Self-Driving Car Project Q&A | Advanced Lane Finding", and there tracker is defined and called as I have provided in my code. However, averaging is not working since all points from frames before are deleted as soon as new tracher instance is being called. Any advice on how to make this work is welcome. Please refer to comment in "tracker.py", line 87). 

However, another simple alternative averaging method is being provided in "video_gen.py", lines 196-212. Together with smoothing, sanity check is being done. Sanity check calculates averaged distance between the lines. If sanity check fails, points from previous frame are taken.

Visualising of found lines is done in "video_gen.py", lines 214-240. Example is given in [sliding_window].

After line pixels have been identified (lines 247-256), calculation of 2nd order polynomial using np.polyfit() function based on centers of sliding windows found in previous step is being performed. Once polynomials for each line are found, they are being plotted on the original, unwarped image. Example is given in [polynomial_result]. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature is calculated in lines 281-287. Calculation is done based on middle radius - polynom coefficents of left and right lane were averaged. Parameters xm_per_pix and ym_per_pix were used for fransforming the radius values from pixels to meters. Additionally, information about distance from line center and radius information were added to the image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Final result on image example is given in [final_result].

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Link to the video result can be found here: [output_video]. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In first part of the pipeline, better tuning of sobel thresholds, kernel size for the sobel filter and color thresholds would help, but this can also be time consuming. Especially since better judgement can be done with larger testing set representing all problematic situations.

To speed up the pipeline and improve accuracy, search from prior could be implemented.

Another disadvantage was not properly used class Lane or Tracker. Any advice on how to implement this feature is welcome.

Harder challenge videos show that this pipeline is struggleing with more sharp curves. There ROIs for detection (or in this project, perspective transform) could be defined differently. Also more dynamic window search could be used. Furthermore, a bit more complex pipeline using CNNs would perform better than sliding window search in such dynamic situations. 
