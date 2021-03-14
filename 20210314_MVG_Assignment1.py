#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
import numpy as np 
import matplotlib.pylab as plt

# function to mask areas of the image outside of the region of interest; it takes two variables, the image
# and the region of interest vertices; it is designed for an image with one channel, a grey scale image

def roi(img, vertices):
    # define a blank matrix that matches the height and width of the image
    mask = np.zeros_like(img)
    # The function operates on grey scale images which have only one channel
    match_mask_colour = 255
    # mask the entire area of the image outside of the region of interest
    cv2.fillPoly(mask, vertices, match_mask_colour)
    # find the parts of the image where the masked pixel matches
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Function to draw lines on an image passed to the function; it takes two parameters, the image and the line vectors

def draw_lines(img, lines):
# Copy the image
    img = np.copy(img)
# Create a blank image with the same dimentions as the original image; the shape of the image is specified in a tuple
# (height, width, no. of channels); the datatype is unsigned integer 
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
# Loop around the line vectors and draw the lines   
    for line in lines:
# (x1, y1) and (x2, y2) are the coordinates of the start and end of the line
        for x1, y1, x2, y2 in line:
# Draw a line on the blank_image using cv2.line: parameters: the image, the coordinates of the two points, 
# define the colour, and define the thickness       
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255,0), thickness = 4)
# Merge the blank_image with the original image; original image, alpha, second image, beta, gamma         
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
# Return the image with the lines on it
    return img


def process(image):
#    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

# Convert the image to a gray scale image
    g_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    g_scale = cv2.GaussianBlur(g_scale, (11,11), 0)
#    cv2.imshow('g_scale', g_scale)

# define the region of interest, roi
    roi_vertices = [(0, height), (width/2, height/2), (width, height)]
    
# Use Canny edge detection to find the edges in the gray scale image
    canny_image = cv2.Canny(g_scale, 100, 120)


    cropped_image = roi(canny_image, np.array([roi_vertices], np.int32))

# Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(cropped_image, 
                        rho=2, 
                        theta=np.pi/80, 
                        threshold=90, 
                        lines=np.array([]), 
                        minLineLength=90, 
                        maxLineGap=50)

    image_with_lines = draw_lines(image, lines)
    return image_with_lines

# This is the video capture function taking in test_video.mp4
#cap = cv2.VideoCapture(os.path.expanduser('~/Desktop/test_video.mp4'))
#cap = cv2.VideoCapture(os.path.expanduser('~/Desktop/solidWhiteRight.mp4'))
cap = cv2.VideoCapture(os.path.expanduser('~/Desktop/Video_A.mp4'))

# Check if the video frames are available; cap.isOpened() is TRUE (returns a boolean variable) 
# while frames are available
while(cap.isOpened()):
# cap.read() returns two variables
    ret, frame = cap.read()
# Apply the process function to the frame - overwrite the frame with the lines 
    frame = process(frame)
# Show the results 
    cv2.imshow('frame', frame)
# Code to quit from this loop; when someone presses the q key we exit the loop    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release function called on cap variable and destroy all windows
cap.release()
cv2.destroyAllWindows()

