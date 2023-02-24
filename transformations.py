# -*- coding: utf-8 -*-
"""
Python implementation for compression ensembles to quantify the aesthetic complexity of images
See paper: https://arxiv.org/abs/2205.10271
"Compression ensembles quantify aesthetic complexity and the evolution of visual art"
Andres Karjus, Mar Canet Solà, Tillmann Ohm, Sebastian E. Ahnert, Maximilian Schich

Note: Our paper may describe slightly different transformations using R and ImageMagick. 
This version uses Python and OpenCV with optimized transformations which should run much faster.
The specific transformations and total number is abritrary for the method (see paper).

"""

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


exportFolder = "image_export/"

def saveImage(IMG, filename):
  image_path = exportFolder + filename + '_image.png'
  cv2.imwrite(image_path, IMG)

#----------------------------------------------------------------------------------------------
# Methods for transformations
#----------------------------------------------------------------------------------------------
# class dithering:
#   def hist_eq(self,im):
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     cl1 = clahe.apply(im)
#     return cl1
#   def set_pixel(im,x,y,new):
#     im[x,y]=new

#   def stucki(self,im):   # stucki algorithm for image dithering
#     w8= 8/42.0
#     w7=7/42.0
#     w5=5/42.0
#     w4= 4/42.0
#     w2=2/42.0
#     w1=1/42.0
#     width,height=im.shape
#     for y in range(0,height-2):
#       for x in range(0,width-2):
#         old_pixel=im[x,y]
#         if old_pixel<127:
#           new_pixel=0
#         else:
#           new_pixel=255	
#         set_pixel(im,x,y,new_pixel)
#         quant_err=old_pixel-new_pixel
#         set_pixel(im,x+1,y, im[x+1,y] + w7 * quant_err);
#         set_pixel(im,x+2,y, im[x+2,y]+ w5 * quant_err);
#         set_pixel(im,x-2,y+1, im[x-2,y+1] + w2 * quant_err);
#         set_pixel(im,x-1,y+1, im[x-1,y+1] + w4 * quant_err);
#         set_pixel(im,x,y+1, im[x,y+1] + w8 * quant_err);
#         set_pixel(im,x+1,y+1, im[x+1,y+1] + w4 * quant_err);
#         set_pixel(im,x+2,y+1, im[x+2,y+1] + w2 * quant_err);
#         set_pixel(im,x-2,y+2, im[x-2,y+2] + w1 * quant_err);
#         set_pixel(im,x-1,y+2, im[x-1,y+2] + w2 * quant_err);
#         set_pixel(im,x,y+2, im[x,y+2] + w4 * quant_err);
#         set_pixel(im,x+1,y+2, im[x+1,y+2] + w2 * quant_err);
#         set_pixel(im,x+2,y+2, im[x+2,y+2]+ w1 * quant_err);
#     return im


#   def quantize(self, im):  # Floyd-Steinberg METHOD of image dithering
#     for y in range(0,height-1):
#       for x in range(1,width-1):
#         old_pixel=im[x,y]
#         if old_pixel<127:
#           new_pixel=0
#         else:
#           new_pixel=255
#         set_pixel(im,x,y,new_pixel)
#         quant_err=old_pixel-new_pixel
#         set_pixel(im,x+1,y,im[x+1,y]+quant_err*w1)
#         set_pixel(im,x-1,y+1, im[x-1,y+1] +  quant_err*w2 )
#         set_pixel(im,x,y+1, im[x,y+1] +  quant_err * w3 )
#         set_pixel(im,x+1,y+1, im[x+1,y+1] +  quant_err * w4 )
#     return im

#   def process(self,img, save=False):
#     img2=img.copy()
#     width,height,z=img.shape
#     #print img.shape
#     w1=7/16.0
#     #print w1
#     w2=3/16.0
#     w3=5/16.0
#     w4=1/16.0

#     gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blue=img[:,:,0]  #taking the blue channel
#     blue=self.stucki(blue)   #sending it to stucki algorithm
#     blue=self.hist_eq(blue)   #histogram equalising the result  same applies for remaining channels below
#     green=img[:,:,1]
#     green= self.stucki(green)
#     green= self.hist_eq(green)
#     red=img[:,:,2]
#     red= self.stucki(red)
#     red= self.hist_eq(red)
#     image_color = cv2.merge((blue, green, red))  #merging the 3 color channels
#     image_gray1= self.hist_eq(gray)
#     image_gray1= self.stucki(image_gray1)
#     image_gray2= self.stucki(gray)
#     return  image_color,image_gray1, image_gray2

def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree  
    filters = []
    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def apply_filter(img, filters):
  # This general function is designed to apply filters to our image
  # First create a numpy array the same size as our input image
  newimage = np.zeros_like(img)
     
  # Starting with a blank image, we loop through the images and apply our Gabor Filter
  # On each iteration, we take the highest value (super impose), until we have the max value across all filters
  # The final image is returned
  depth = -1 # remain depth same as original image
     
  for kern in filters:  # Loop through the kernels in our GaborFilter
      image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image   
      # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
      np.maximum(newimage, image_filter, newimage)
  return newimage
  
# ----------------------------------------------------------------------------------------------------------------------
# Transformation methods
#----------------------------------------------------------------------------------------------

def simpleBlur(img, save=False):
  simpleBlur = cv2.blur(img,(5,5))
  if save:
    saveImage(simpleBlur, "simpleBlur")
  return simpleBlur

def gaussianBlur(img, save=False):
  gaussianBlur = cv2.GaussianBlur(img,(25, 25),0)
  if save:
    saveImage(gaussianBlur, "gaussianBlur")
  return gaussianBlur

def hardBlur(img, save=False):
  hardBlur = cv2.blur(img,(30,30))
  if save:
    saveImage(hardBlur, "hardBlur")
  return hardBlur

def convertGreyscale(img, save=False):
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  if save:
    saveImage(gray_img, "convertGreyscale")
  return gray_img

def cannyEdgeDetection(img, save=False):
  canny = cv2.Canny(img, threshold1=30, threshold2=100)
  if save:
    saveImage(canny, "cannyEdgeDetection")
  return canny

def sobelEdgeDetection(gray_img, save=False):
  horizontal = cv2.Sobel(gray_img,0,1,0,cv2.CV_64F)
  # the thresholds are like 
  # (variable,0,<x axis>,<y axis>,cv2.CV_64F)
  vertical = cv2.Sobel(gray_img,0,0,1,cv2.CV_64F)
  # DO the Bitwise operation
  bitwise_Or = cv2.bitwise_or(horizontal,vertical)
  if save:
    saveImage(bitwise_Or, "sobelEdgeDetection")
  return bitwise_Or

def laplacianDetection(gray_img, save=False):
  # Make Laplacian Function
  lappy=cv2.Laplacian(gray_img,cv2.CV_64F)
  if save:
    saveImage(lappy, "laplacianDetection")
  return lappy

def sobelplusblurEdgeDetection(gray_img, save=False):
  blur= cv2.GaussianBlur(gray_img, (15,15) , 0 )
  horizontal = cv2.Sobel(blur,0,1,0,cv2.CV_64F)
  # the thresholds are like 
  # (variable,0,<x axis>,<y axis>,cv2.CV_64F)
  vertical = cv2.Sobel(gray_img,0,0,1,cv2.CV_64F)
  # DO the Bitwise operation
  bitwise_Or = cv2.bitwise_or(horizontal,vertical)
  if save:
    saveImage(bitwise_Or, "sobelplusblurEdgeDetection")
  return bitwise_Or

def gaborKernel(img, save=False):
  gfilters = create_gaborfilter()
  gaborKernel = apply_filter(img, gfilters)
  if save:
    saveImage(gaborKernel, "gaborKernel")
  return gaborKernel

def gaborGreyKernel(gray_img, save=False):
  gfilters = create_gaborfilter()
  gaborKernel = apply_filter(gray_img, gfilters)
  if save:
    saveImage(gaborKernel, "gaborGreyKernel")
  return gaborKernel

def filter2DKernel1(gray_img, save=False):
  # Defining the kernel of size 3x3
  kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
  ])
  filter2DKernel1 = cv2.filter2D(gray_img, -1, kernel)
  if save:
    saveImage(filter2DKernel1, "filter2DKernel1")
  return filter2DKernel1

def embossFilter(gray_img, save=False):
  kernel = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
  ])
  embossFilter = cv2.filter2D(gray_img, -1, kernel)
  if save:
    saveImage(embossFilter, "embossFilter")
  return embossFilter

def sobelFilter(gray_img, save=False):
  # Defining the Sobel kernel of size 3x3
  kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
  ])
  sobelFilter = cv2.filter2D(gray_img, -1, kernel)
  if save:
    saveImage(sobelFilter, "sobelFilter")
  return sobelFilter

def boxFilter(img, save=False):
  # Kernal or Convolution matrix for Box BLue Filter
  kernal = np.ones((5, 5), np.uint8) / 25
  output = cv2.filter2D(img, -1, kernal)
  # Low pass filters implementation
  box_filter = cv2.boxFilter(img, -1, (31, 31))
  if save:
    saveImage(box_filter, "boxFilter")
  return box_filter

def arcCosine(img, save=False):
  imgFloat = np.float32(img)/255.0  
  cos = np.arccos(imgFloat)
  arcCos = np.uint8(cos*255.0)  
  if save:
    saveImage(arcCos, "arcCosine")
  return arcCos

def powerTen(img, save=False):
  imgFloat = np.float32(img)/255.0
  p10 = np.power(imgFloat,10)
  powerTen = np.uint8(p10*255.0) 
  if save:
    saveImage(powerTen, "powerTen")
  return powerTen

def squareRoot(img, save=False):
  imgFloat = np.float32(img)/255.0  
  sqr = np.sqrt(imgFloat)
  squareRoot = np.uint8(sqr*255.0) 
  if save:
    saveImage(squareRoot, "squareRoot")
  return squareRoot

def brightness(img, save=False):
  brighten = np.int16(img)
  brighten = brighten + 200
  brighten = np.clip(brighten, 0, 255)
  brighten = np.uint8(brighten)
  if save:
    saveImage(brighten, "brightness")
  return brighten

def saturation(HLS, save=False):
  HLS[:,:,1] = 255
  saturated = cv2.cvtColor(HLS, cv2.COLOR_HSV2BGR)
  if save:
    saveImage(saturated, "saturation")
  return saturated

def meansDenoising(img, strength, save=False):
  denoised = cv2.fastNlMeansDenoisingColored(img,None,strength,strength,7,21)
  if save:
    saveImage(denoised, "meansDenoising"+str(strength))
  return denoised

def kMeansQuantize(img,  save=False, nCluster=3):
  (h, w) = img.shape[:2]
  image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  image = image.reshape((image.shape[0] * image.shape[1], 3))
  clt = MiniBatchKMeans(n_clusters = nCluster)
  labels = clt.fit_predict(image)
  quant = clt.cluster_centers_.astype("uint8")[labels]
  quant = quant.reshape((h, w, 3))
  quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
  if save:
    saveImage(quant, "kMeansQuantize"+str(nCluster))
  return quant

def OTSUQuantize(gray_img,  save=False):
  ret, OTSU = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  if save:
    saveImage(OTSU, "OTSUQuantize")
  return OTSU

# magnitude spectrum filter (not working yet)
'''
def fftFilter(img, save=False):
  f = np.fft.fft2(img)
  fshift = np.fft.fftshift(f)
  magnitude_spectrum = 20*np.log(np.abs(fshift))
  if save:
    saveImage(magnitude_spectrum, sys._getframe().f_code.co_name)
 
  return get_ratio(compress(magnitude_spectrum,format),baseline)
'''

def ditheringFilter(img, save=False):
  myDithering = dithering()
  image_color,image_gray1, image_gray2 = myDithering.process(img)
  if save:
    saveImage(image_color, "ditheringFilter")
  return image_color

def distanceTransform(gray_img, save=False):
  ret, thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  # noise removal
  kernel = np.ones((2,2),np.uint8)
  #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
  closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
  # sure background area
  sure_bg = cv2.dilate(closing,kernel,iterations=3)
  # Finding sure foreground area
  dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
  ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
  if save:
    saveImage(sure_fg, "distanceTransform")
  return sure_fg

def pixelate(IMG,  factor, save=False, method=3):
  h,w,_ = IMG.shape
  dim = (int(w/factor),int(h/factor))
  resized = cv2.resize(IMG, dim, interpolation=method)
  resized = cv2.resize(resized, (w,h), interpolation=method)
  if save:
    saveImage(resized, "pixelate"+str(factor)+"_"+str(method))
  return resized

def bwQuantizeThreshold(gray_img,  save=False):
  ret,thresh = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
  if save:
    saveImage(thresh, "bwQuantizeThreshold")
  return thresh

def floodFill(IMG,  start, save=False):
  h, w = IMG.shape[:2]
  flooded = IMG.copy()
  # slide through middle of upper third
  if start.startswith("H_"):
    wGap = int(h/12)
    if start == "H_upperThird":
      x = int(h/6)
    elif start == "H_lowerThird":
      x = int(h-(h/6))
    elif start == "H_horizontal":
      x = int(h/2)
    for y in range(wGap,w-wGap):
      cv2.floodFill(flooded, None, (y,x), (255, 255, 255), loDiff=(1, 1, 1, 1), upDiff=(1, 1, 1, 1))
  elif start.startswith("V_"):
    hGap = int(h/12)
    if start == "V_leftThird":
      y = int(w/6)
    elif start == "V_rightThird":
      y = int(w-(w/6))
    elif start == "V_vertical":
      y = int(w/2)
    for x in range(hGap,h-hGap):
      cv2.floodFill(flooded, None, (y,x), (255, 255, 255), loDiff=(1, 1, 1, 1), upDiff=(1, 1, 1, 1))
  else:
    if start == "middle":
      seed = (int(w/2),int(h/2))
    elif start == "upperLeftThird":
      seed = (int(w/3),int(h/3))
    elif start == "upperRightThird":
      seed = (int(w-(w/3)),int(h/3))
    elif start == "lowerLeftThird":
      seed = (int(w/3),int(h-(h/3)))
    elif start == "lowerRightThird":
      seed = (int(w-(w/3)),int(h-(h/3)))
    cv2.floodFill(flooded, None, seed, (255, 255, 255), loDiff=(1, 1, 1, 1), upDiff=(1, 1, 1, 1))
  if save:
    saveImage(flooded, "floodFill_"+start)
  return flooded

def distanceTransformB(type,gray_img, format, baseline,save=False):
  _, threshold = cv2.threshold(gray_img, 35, 255, cv2.THRESH_BINARY)
 
  # Calculate the distance transform
  if type=='1':
    distTransform_filter = cv2.distanceTransform(threshold, cv2.DIST_C, 3)
  elif type=='2':
    distTransform_filter = cv2.distanceTransform(threshold, cv2.DIST_L1, 3)
  else:
    distTransform_filter = cv2.distanceTransform(threshold, cv2.DIST_L2, 3)

  if save:
    image_path ='distanceFilter'+type+'_image.png'
    imageTransformed.append(image_path) 
    cv2.imwrite(image_path,distTransform_filter)
  return get_ratio(compress(distTransform_filter,format),baseline)

def houghLinesP(gray_img, format, baseline,save=False):
  houghLinesP_img = gray_img.copy()
  #houghLinesP_img = cv2.rectangle(gray_img,(0,0),(gray_img.shape[1],gray_img.shape[0]),(0,0,0),thickness=-1)
  #houghLinesP_img = np.zeros((gray_img.shape[0], gray_img.shape[1], 1), dtype = "uint8")
  #houghLinesP_img = cv2.cvtColor(gray_img2,cv2.COLOR_GRAY2RGB)
  houghLinesP_img = cv2.rectangle(houghLinesP_img,(0,0),(houghLinesP_img.shape[1],gray_img.shape[0]),(0,0,0),thickness=-1)
  canny = cv2.Canny(gray_img, threshold1=30, threshold2=100)
  lines = cv2.HoughLinesP(canny,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
  #houghLinesP_img = gray_img
  for line in lines:
      x1,y1,x2,y2 = line[0]
      cv2.line(houghLinesP_img,(x1,y1),(x2,y2),(255,255,255),2)
  if save:
    image_path ='houghLinesP_image.png'
    imageTransformed.append(image_path)
    cv2.imwrite(image_path, houghLinesP_img)
  return get_ratio(compress(houghLinesP_img,format),baseline)

def adaptiveThreshold(gray_img, format, baseline,save=False):
  img_grey = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
  adaptiveThreshold_filter = cv2.adaptiveThreshold(img_grey ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
  if save:
    image_path ='adaptiveThresholdFilter_image.png'
    imageTransformed.append(image_path) 
    cv2.imwrite(image_path,adaptiveThreshold_filter)
  return get_ratio(compress(adaptiveThreshold_filter,format),baseline)

# def floodFill(IMG,  start, save=False):
#   h, w = IMG.shape[:2]
#   flooded = IMG.copy()
#   if start == "corners":
#     seeds = [(5,5),(w-5,5),(5,h-5),(w-5,h-5)]
#   elif start == "center":
#     seeds = [(int(w/2),int(h/2))] 
#   elif start == "upper":
#     seeds = [(int(w/2),5)]
#   elif start == "lower":
#     seeds = [(int(w/2),h-5)] 
#   for seed in seeds:
#     cv2.floodFill(flooded, None, seed, (255, 255, 255), loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
#   if save:
#     saveImage(flooded, "floodFill_"+start)
#   return get_ratio(compress(flooded,format),baseline)
