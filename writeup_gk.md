## **Finding the Lane Lines on the Road**
When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

## Getting Started
To develop a simple pipeline for finding the lane lines in an image, then apply this pipeline to a full video feed.

We will be leveraging some of the below popular algorithms and packages:
* **OpenCV**            - Algorithms for Computer Vision 
* **NumPy**             - Algorithms for Scientifc computation
* **SciPy**             - Algorithms for Scientifc computation
* **Matplotlib**        - A Python 2D plotting library
* **Pyplot**            - A Python library for interactive plot generation
* **Gaussian blur**     - Algorithms for Image processing
* **Canny**             - Algorithm for edge detection in an image
* **Hough transform**   - An  algorithm to identify location of lane lines on the Road
* **Linear regression** - TO find the best relationship between a group of lane points

## Prerequisites
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
* Install all the required packages
```
import matplotlib.pyplot as plt
import numpy as np
import cv2
```
* Jupyter Notebook to build and test the pipeline
```
# test the enviornment using jupyter notebook
jupyter notebook P1.ipynb

```
![Figure 1.1: An Unprocess Frame](https://user-images.githubusercontent.com/12469124/34318397-990c2444-e794-11e7-8187-bd2c60a801d6.jpeg) 

Figure 1.1: An Unprocess Frame

![Figure 1.2: A frame with lanes automatically indicated](https://user-images.githubusercontent.com/12469124/34318399-a053c3b0-e794-11e7-9398-8f02108bdf5b.jpeg)

Figure 1.2: A frame with lanes automatically indicated

## Lane Detection Pipeline
Below are the steps involved in my Lane Detection Pipeline.

### 1. Image cleanup and noise removal
I used Gaussian blur algorithm to clean up the image. 
Gaussian blur algorithm is applied to remove the noise and tiny details from the image such as distant objects that are irrelevant for our purpose

```
def image_cleanup(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
```

### 2. Convert the image to grayscale
convert the image to grayscale before isolating the region of interest
highlight pixels with a higher brightness value, including the ones defining marking lanes

I used cv2.cvtColor a Grayscale Image Convertor function with parameters - image and cv2.COLOR_BGR2GRAY
```
def discard_colors(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### 3. Detect the edges
I applied the Canny edge detection algorithm which detects edges on a picture by looking for quick changes in color between a pixel and its neighbors. 
The blur and grayscale step will help make the main lane lines stand out. 
The result will give a black and white image. 

I used cv2.Canny function with parameters Image, low threshold, high threshold
```
def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)
```

### 4. Masking the region of interest
```
def region_of_interest(image, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(image)
   
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are non-zero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
xsize = img.shape[1]
ysize = img.shape[0]
dx1 = int(0.0725 * xsize)
dx2 = int(0.425 * xsize)
dy = int(0.6 * ysize)

# calculate vertices for region of interest
vertices = np.array([[(dx1, ysize), (dx2, dy), (xsize - dx2, dy), (xsize - dx1, ysize)]], dtype=np.int32)
image = region_of_interest(image, vertices)
```

### 5. Identify the location of lane lines on the road
The Hough transform algorithm is applied-
to extracts all the lines passing through each of our edge points
group them by similarity 

The HoughLinesP function in OpenCV returns an array of lines organized by endpoints (x1, x1, x2, x2).
```
def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines
rho = 0.8
theta = np.pi/180
threshold = 25
min_line_len = 50
max_line_gap = 200

lines = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)
```
### 6. Separate  the lines
The Hough transformation gives us back multiple lines.
So we need to pick  only two distinct lanes - left and right lane markeres for our car to drive in between. 

The lines are organised by its slope. Positive slopes are for the right lane and negative slopes are for the left lane.


```
def separate_lines(lines):
    right = []
    left = []
    for x1,y1,x2,y2 in lines[:, 0]:
        m = get_slope(x1,y1,x2,y2)
        if m >= 0:
            right.append([x1,y1,x2,y2,m])
        else:
            left.append([x1,y1,x2,y2,m])
    return right, left
right_lines, left_lines = separate_lines(lines)
```

### 7. Reject the outliers
reject lines with unacceptable slopes that throw off the intended slope of each line.

```
def reject_outliers(data, cutoff, thresh=0.08):
    data = np.array(data)
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    m = np.mean(data[:, 4], axis=0)
    return data[(data[:, 4] <= m+thresh) & (data[:, 4] >= m-thresh)]
if len(right_lines) != 0 and len(left_lines) != 0:
    right = reject_outliers(right_lines,  cutoff=(0.45, 0.75))
    left = reject_outliers(left_lines, cutoff=(-0.85, -0.6))
```

### 8. Merge left and right lanes  using linear regression.

```
def lines_linreg(lines_array):
    x = np.reshape(lines_array[:, [0, 2]], (1, len(lines_array) * 2))[0]
    y = np.reshape(lines_array[:, [1, 3]], (1, len(lines_array) * 2))[0]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    x = np.array(x)
    y = np.array(x * m + c)
    return x, y, m, c

x, y, m, c = lines_linreg(lines)
# This variable represents the top-most point in the image where we can reasonable draw a line to.
min_y = np.min(y)
# Calculate the top point using the slopes and intercepts we got from linear regression.
top_point = np.array([(min_y - c) / m, min_y], dtype=int)
# Repeat this process to find the bottom left point.
max_y = np.max(y)
bot_point = np.array([(max_y - c) / m, max_y], dtype=int)

```

### 9. Span the Lines
Using some simple geometry (y = mx + b), calculate extrema. 
I used the result of the linear regression to extrapolate to those extrema. 
I then extended the left and right lines off across the image and clip the line using our previous region of interest.

```
def extend_point(x1, y1, x2, y2, length):
    line_len = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    x = x2 + (x2 - x1) / line_len * length
    y = y2 + (y2 - y1) / line_len * length
    return x, y
x1e, y1e = extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1], -1000) # bottom point
x2e, y2e = extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1],  1000) # top point
# return the line.
line = np.array([[x1e,y1e,x2e,y2e]])
return np.array([line], dtype=np.int32)

```

### 10. Draw the lines and return the final image
The final step is to superimpose the left and right lines onto the original image to visually validate the correctness and accuracy of our pipeline implementation.

```
def weighted_image(image, initial_image, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_image, α, image, β, λ)
line_image = np.copy((image)*0)
draw_lines(line_image, lines, thickness=3)
line_image = region_of_interest(line_image, vertices)
final_image = weighted_image(line_image, image)
return final_image
``` 

## Potential shortcomings with the current pipeline
One potential shortcoming would be that  we decided to discard color information and rely exclusively on pixel brightness to detect lane marking on the road. It works well during daylight and with a simple terrain but  the lane detection accuracy might drop significantly in less ideal conditions.



## possible improvements to the pipeline

A possible improvement would be to transform the original image from RGB colorspace to HSV colorspace.
we can filter out pixels which are not the given color of the lanes. 
Using hue and saturation values, not how dark a pixel is, will ensure lines of a given color are more easily detected in shaded or low-contrast regions.
Create two color filters that will extract the whites and yellows in the image and apply them to turn black any other pixels.

```
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
