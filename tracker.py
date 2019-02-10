import numpy as np
import cv2

class tracker():	
		
	def __init__(self, Mywindow_width, Mywindow_height, Mymargin, Myminpix = 50, My_ym = 1, My_xm = 1, Mysmooth_factor = 15):
		#list of all past left and right centers
		self.recent_centers = []
		#print('initialized in self', self.recent_centers)
		
		#width of convolving window (?)
		self.window_width = Mywindow_width
		
		#height of sliding windows
		self.window_height = Mywindow_height
		
		#pixel distance in both directions to slide (range around previous polynomial to search for pixels)
		self.margin = Mymargin
		
		#minimum number of pixels needed to recenter the window
		self.minpix = Myminpix
		
		#meter per pixel in y
		self.ym_per_pix = My_ym

		#meter per pixel in x
		self.xm_per_pix = My_xm
		
		#smoothing (averaging over last n frames)
		self.smooth_factor = Mysmooth_factor
		
	def find_window_centroids(self, warped):
		#print(self)
		##__________Find window centroids for each frame__________
		window_width = self.window_width # for convolution
		window_height = self.window_height # for determining layer height
		margin = self.margin # used to find centroid based on previous center
		minpix = self.minpix # min. pixels required to find within window
		
		#initialize centroids and sliding window
		window_centroids = [] # storing centers
		window = np.ones(window_width)
		
		#finding initial starting point of left and right curve based on bottom quarter
		l_sum = np.sum(warped[warped.shape[0]//4*3:, :warped.shape[1]//2], axis = 0)
		l_center = np.argmax(np.convolve(window, l_sum)) - window_width//2
		
		r_sum = np.sum(warped[warped.shape[0]//4*3:, warped.shape[1]//2:], axis = 0)
		r_center = np.argmax(np.convolve(window, r_sum)) - window_width//2 + warped.shape[1]//2
		
		#append to window centroids center list for this frame
		l_center = int(l_center)
		r_center = int(r_center)
		window_centroids.append((l_center,r_center))
		
		#looping through each layer of window_height, ignoring most bottom one
		for level in range(1, warped.shape[0]//window_height):
			#sum the layer along x axis
			image_layer = np.sum(warped[(warped.shape[0] - (level+1)*window_height):(warped.shape[0] - (level)*window_height),:], axis = 0)
			#convolve summed layer with convolution window
			conv_signal = np.convolve(window,image_layer)
			#offset caused by convolutional signal reference
			offset = window_width/2
			
			#finding best centroid based on previous center within margin
			l_min_index = int(max(l_center+offset-margin,0))
			#print('l_center+offset-margin', l_center+offset-margin, 'int(max(l_center+offset-margin,0))', int(max(l_center+offset-margin,0)))
			l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
			## sanity check left (minimum pixels discovered)
			#if enough pixels found, take new center where maximum pixels are. If not, keep old center.
			if (sum(image_layer[(l_center - margin):(l_center+margin)]) > minpix):
				l_center = np.argmax(conv_signal[l_min_index:l_max_index]) - offset + l_min_index
			r_min_index = int(max(r_center+offset-margin,0))
			r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
			
			## sanity check right (minimum pixels discovered)
			#if enough pixels found, take new center where maximum pixels are. If not, keep old center.
			if (int(sum(image_layer[(r_center - margin):(r_center+margin)])) > minpix):
				r_center = np.argmax(conv_signal[r_min_index:r_max_index]) - offset + r_min_index
			l_center = int(l_center)
			r_center = int(r_center)
			
			#append new centers
			window_centroids.append((int(l_center),int(r_center)))
			
		#__________Message for reviewer______________________	
		#append centroids of the frame to class tracker, should be used for averaging centers -- unclear how this can stay saved if with every new image new class tracker is called. Currently implemented in main file video_gen.py, lines 
		#_____________________________________________________
		
		self.recent_centers.append(window_centroids)
		#return averaged centers on past smooth_factor values, to avoid sudden jumps from frame to frame
		return np.average(self.recent_centers[-self.smooth_factor:],axis = 0)