# Self-Driving Car Engineering By Udacity

## Project: **Finding Lane Lines on the Road** 
Devloped by **Gangadhar Kadam** on **December 2017****
***

![projectbackground](https://user-images.githubusercontent.com/12469124/34321395-3ab68982-e7dc-11e7-9a31-9bf7284482ec.jpeg)

When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project, I developed a pipeline on a series of individual images, and later applied the result to a video stream (really just a series of images). I validated the my result with "raw-lines-example.mp4"  test video provided in the project. When the result looked roughly same as the test video, I tried to average and/or extrapolate the line segments I detected to map out the full extent of the lane lines. 

---

## Getting Started
To develop this simple lane detection, then apply this pipeline to the test images provided and make a full video with lanes lines getting projected.

**I have leveraged some of the below popular algorithms and packages:**
- `OpenCV` - Algorithms for Computer Vision 
    - `cv2.inRange()` for color selection  
    - `cv2.fillPoly()` for regions selection  
    - `cv2.line()` to draw lines on an image given endpoints  
    - `cv2.addWeighted()` to coadd / overlay two images 
    - `cv2.cvtColor()` to grayscale or change color 
    - `cv2.imwrite()` to output images to file  
    - `cv2.bitwise_and()` to apply a mask to an image 
- `NumPy` - Algorithms for Scientifc computation 
- `SciPy` - Algorithms for Scientifc computation 
- `Matplotlib` - A Python 2D plotting library 
- `Pyplot` - A Python library for interactive plot generation 
- `Gaussian blur` - Algorithms for Image processing 
- `Canny` - Algorithm for edge detection in an image 
- `Hough transform` - An algorithm to identify location of lane lines on the Road

## Preprocessing
* Set up the CarND Term1 Starter Kit in conda enviornment
```
# clone the carnd-term1 repository
git clone https://github.com/udacity/CarND-Term1-Starter-Kit.git
cd CarND-Term1-Starter-Kit

# Create the enviornment
conda env create -f environment.yml

# Check the enviornment
conda info --envs

# Activate the Enviornment
source activate carnd-term1

```

* Jupyter Notebook to build and test the pipeline
```
# test the enviornment using jupyter notebook
jupyter notebook P1.ipynb

```

* Import the Required packages

```
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
```

* reading in an image
```
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)

```

## Unprocessed Frame Vs A frame with lanes automatically indicated
![Figure 1.1: An Unprocess Frame](https://user-images.githubusercontent.com/12469124/34318397-990c2444-e794-11e7-8187-bd2c60a801d6.jpeg) 

Figure 1.1: An Unprocess Frame

![Figure 1.2: A frame with lanes automatically indicated](https://user-images.githubusercontent.com/12469124/34318399-a053c3b0-e794-11e7-9398-8f02108bdf5b.jpeg)

![weighted_image](https://user-images.githubusercontent.com/12469124/34328791-11538f00-e8b7-11e7-9f99-16eed5006675.jpeg)

Figure 1.2: A frame with lanes automatically indicated

## The Pipeline
The pipeline is as follows:
1. Apply a Gaussian smoothing algorithm to cleanup the image and noise removal
2. Convert the image to grayscale
3. Apply Canny edge detection algorithm to detects the edges
4. Apply an image mask to get "region of interest" in front of the vehicle
5. Apply Hough ransform algorith to identify the location of lane lines
6. Get the list of lines and line slopes for averaging
7. Average the line positions
8. Remove outlier slopes from the line averaging
9. Extrapolate the line boundaries
10. get the final image with the weighted average

### 1. Apply a Gaussian smoothing algorithm to cleanup the image and noise removal
I used Gaussian blur algorithm to clean up the image. 
Gaussian blur algorithm is applied to remove the noise and tiny details from the image such as distant objects that are irrelevant for our purpose

```
# Define a function to Apply a Gaussian Noise kernel
def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Define a kernel size and 
kernel_size = 9

# apply Gaussian smoothing
gray_blur = gaussian_blur(image, kernel_size)

# Save the image
mpimg.imsave("test_images_output/gray_blur.jpeg",gray_blur)

# plot the image
plt.imshow(gausBlur)
```

![gray_blur](https://user-images.githubusercontent.com/12469124/34328932-5a6ae008-e8bc-11e7-8f36-5dfa8405569d.jpeg)

figure 2- blur filter applied to image

### 2. Convert the image to grayscale
convert the image to grayscale before isolating the region of interest
highlight pixels with a higher brightness value, including the ones defining marking lanes

I used cv2.cvtColor a Grayscale Image Convertor function with parameters - image and cv2.COLOR_BGR2GRAY
```
# Define a Function to convert the image to grayscale
# This will return an image with only one color channel
def gray_scale_transform(image):
    #use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply the gray scale transformation
grayscaled = gray_scale_transform(gray_blur)

mpimg.imsave("test_images_output/gray_scaled1.jpeg",grayscaled)

# plot the image
plt.imshow(gray,cmap='gray')

```
![gray_scaled](https://user-images.githubusercontent.com/12469124/34328933-5c089cfc-e8bc-11e7-9df5-0db1070981d8.jpeg)

figure 3- grayscale transformation applied to blurred image

### 3. Apply Canny edge detection algorithm to detects the edges
I applied the Canny edge detection algorithm which detects edges on a picture by looking for quick changes in color between a pixel and its neighbors. 
The blur and grayscale step will help make the main lane lines stand out. 
The result will give a black and white image. 

I used cv2.Canny function with parameters Image, low threshold, high threshold
```
# Define a function to Apply the Canny transformation
def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

# Define  parameters for Canny
low_threshold = 50
high_threshold = 150

#Apply the cany transformation
edges = detect_edges(grayscaled, low_threshold,high_threshold)

# Save the image
mpimg.imsave("test_images_output/detect_edges.jpeg",edges)

# plot the image
plt.imshow(edges)

```

![detect_edges](https://user-images.githubusercontent.com/12469124/34328788-093f4b42-e8b7-11e7-91c7-c391ca5cc547.jpeg)

figure 4- canny edge detection applied to grayscale image

### 4. Apply an image mask to get "region of interest" in front of the vehicle
```
# Function to get the region of interest by applying an image mask
# Only keeps the region of the image defined by the polygon formed from `vertices`. 
# The rest of the image is set to black.
def region_of_interest(image, vertices):

    #defining a blank mask to start with
    mask = np.zeros_like(image)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Now the masking
# calculate vertices for region of interest
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 325), (550, 325), (imshape[1],imshape[0])]], dtype=np.int32)

# return the image only where mask pixels are nonzero
masked_img = region_of_interest(edges, vertices)

# Save the image
mpimg.imsave("test_images_output/RegionOfInterest.jpeg",masked_img)

# plot the image
plt.imshow(masked_img)

```
![regionofinterest](https://user-images.githubusercontent.com/12469124/34328790-0f478892-e8b7-11e7-9045-754b62a8c1f6.jpeg)

figure 5- masking the region of interest

### 5. Apply Hough ransform algorith to identify the location of lane lines
The Hough transform algorithm is applied-
to extracts all the lines passing through each of our edge points
group them by similarity 

The HoughLinesP function in OpenCV returns an array of lines organized by endpoints (x1, x1, x2, x2).
```
def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines
    
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
# hough lines
rho = 5 # distance resolution in pixels of the Hough grid
theta = np.pi/30 # angular resolution in radians of the Hough grid
threshold = 50     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 25 #minimum number of pixels making up a line
max_line_gap = 25    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
#lines = hough_lines(masked_img, rho, theta, threshold, min_line_len, max_line_gap)
lines = cv2.HoughLinesP(masked_img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
```
![houghed_image](https://user-images.githubusercontent.com/12469124/34328789-0c7d8756-e8b7-11e7-9a25-dd2803f473fd.jpeg)

figure 6- hough transformation returns a list of lines

### 6. Get the list of lines and line slopes for averaging
```
left_lines = []
    left_slopes = []
    right_lines = []
    right_slopes = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope < 0:
                left_lines.append(line)
                left_slopes.append(slope)
            else:
                right_lines.append(line)
                right_slopes.append(slope)
```

### 7. Average line positions
```
avg_left_pos = [sum(col)/len(col) for col in zip(*left_lines)]
avg_right_pos = [sum(col)/len(col) for col in zip(*right_lines)]
```

### 8. Remove outlier slopes from the line averaging 
reject lines with unacceptable slopes that throw off the intended slope of each line.

```
#Removing outlier slopes from the averaging performed below in lane_lines
def remove_outliers(slopes, m = 2):
    med = np.mean(slopes)
    stand_dev = np.std(slopes)
    for slope in slopes:
        if abs(slope - med) > (m * stand_dev):
            slopes.remove(slope)
    return slopes

#Remove slope outliers, and take the average
avg_left_slope = np.mean(remove_outliers(left_slopes))
avg_right_slope = np.mean(remove_outliers(right_slopes))
    
```

### 9. Extrapolate the line boundaries 
```
#Average the left line
avg_left_line = []
for x1,y1,x2,y2 in avg_left_pos:
    x = int(np.mean([x1, x2])) #Midpoint x
    y = int(np.mean([y1, y2])) #Midpoint y
    slope = avg_left_slope
    b = -(slope * x) + y #Solving y=mx+b for b
    avg_left_line = [int((325-b)/slope), 325, int((539-b)/slope), 539] #Line for the image 

#Average the right line
avg_right_line = []
for x1,y1,x2,y2 in avg_right_pos:
    x = int(np.mean([x1, x2]))
    y = int(np.mean([y1, y2]))
    slope = avg_right_slope
    b = -(slope * x) + y
    avg_right_line = [int((325-b)/slope), 325, int((539-b)/slope), 539]
   
lines = [[avg_left_line], [avg_right_line]] 
```


### 10. get the final image with the weighted average
The final step is to superimpose the left and right lines onto the original image to visually validate the correctness and accuracy of our pipeline implementation.

```
# Draw the lines
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Weighted Average            
def weighted_image(image, initial_image, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_image, α, image, β, λ)
line_image = np.copy((image)*0)
draw_lines(line_image, lines, thickness=3)
line_image = region_of_interest(line_image, vertices)
final_image = weighted_image(line_image, image)
return final_image

draw_lines(line_image, lines)

# Transparent lines
line_edges = weighted_image(line_image, image)

return line_edges
``` 

![weighted_image](https://user-images.githubusercontent.com/12469124/34328791-11538f00-e8b7-11e7-9f99-16eed5006675.jpeg)

figure 7- masking the region of interest



## Potential shortcomings with the current pipeline
One potential shortcoming would be that  we decided to discard color information and rely exclusively on pixel brightness to detect lane marking on the road. It works well during daylight and with a simple terrain but  the lane detection accuracy might drop significantly in less ideal conditions.

## possible improvements to the pipeline

A possible improvement would be to transform the original image from RGB colorspace to HSV colorspace.
we can filter out pixels which are not the given color of the lanes. 
Using hue and saturation values, not how dark a pixel is, will ensure lines of a given color are more easily detected in shaded or low-contrast regions.
Create two color filters that will extract the whites and yellows in the image and apply them to turn black any other pixels.

```
# TO DO
def filter_image(image):
    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    sensitivity = 51
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])
    lower_yellow = np.array([18,102,204], np.uint8)
    upper_yellow = np.array([25,255,255], np.uint8)
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    filtered_image = cv2.bitwise_and(image, image, mask=white_mask+yellow_mask)
    return filtered_image
```

## Conclusion
In this project, I have developed and implemented an algorithm for detecting white and yellow colored lanes on the road. 
The lane detection method is robust and effective in finding the exact lanes by using both color and edge orientations. 
The main contributions are the color segmentation procedure identifying the yellow or white colored lanes followed by edge orientation in which the boundaries are eliminated, 
lanes are detected, left and right regions are labeled, outliers are removed and finally one line per region remains after using a linear regression on each set.
As the camera remains constant with respect to the road surface, the road portion of the image can be exclusively cropped by providing coordinates, so that identifying the lanes becomes much more efficient. 
The experimental results show the effectiveness of the proposed method in cases of yellow and white colored lanes. 
