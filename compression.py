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
from PIL import Image as PIL_Image
from io import BytesIO
import statistics as stats
import transformations as trans

#----------------------------------------------------------------------------------------------
# Support methods
#----------------------------------------------------------------------------------------------
def ratio(a, b):
  a = float(a)
  b = float(b)
  if b == 0:
    return a
  return ratio(b, a % b)

def get_ratio(a, b):
  r = ratio(a, b)
  return float((a/r) / (b/r))
    
# using PIL to encode image in memory and get size
def compress(imageOpenCV,format,quality=None):
  # if is a openCv image convert to np array
  if (type(imageOpenCV) is np.ndarray):
    imageRotated = cv2.rotate(imageOpenCV,cv2.ROTATE_90_CLOCKWISE)
    imageRotated = PIL_Image.fromarray(imageRotated)
    image = PIL_Image.fromarray(imageOpenCV)
  else:
    ### ToDO: rotate PIL image
    image = imageOpenCV
  # get mean of the file sizes of original image and 90 degree rotated image
  outputOriginal = BytesIO()
  output90Degree = BytesIO()
  if quality:
    image.save(outputOriginal, format=format, quality=quality)
    imageRotated.save(output90Degree, format=format, quality=quality)
  else:
    image.save(outputOriginal, format=format) 
    imageRotated.save(output90Degree, format=format) 
  return (len(outputOriginal.getvalue())+len(output90Degree.getvalue()))/2

def prepareImageVersions(IMG):
  resizedImages = {"100": IMG}
  baselines = {"100": compress(IMG, "bmp")}
  h,w,_ = IMG.shape
  for size in ["40","20","10"]:
    fac = int(size)/100
    dim = (int(w*fac),int(h*fac))
    resized = cv2.resize(IMG, dim, interpolation= cv2.INTER_LINEAR)
    resizedImages[size] = resized
    baselines[size] = compress(resized, "bmp")
  vector = {}
  for format in ["png","gif"]:
    for size in resizedImages:
      img_size = compress(resizedImages[size],format)
      vector["compress_" + format + "_" + size] = get_ratio(img_size, baselines[size])
      baselines[format + "_" + size] = img_size
  for size in resizedImages:
      vector["compress_jpg0_" + size] = get_ratio(compress(resizedImages[size],"jpeg",quality=0), baselines[size])
  for size in resizedImages:
      vector["compress_jpg100_" + size] = get_ratio(compress(resizedImages[size],"jpeg",quality=100), baselines[size])
  # prepate grayscale images for reuse
  return vector, resizedImages, baselines 


#----------------------------------------------------------------------------------------------
# Complexity transformations
#----------------------------------------------------------------------------------------------
exportFolder = "image_export/"

def compressComplexity(img_100, save=False):
  vector, resizedImages, baselines = prepareImageVersions(img_100)
  img_40 = resizedImages["40"]
  gray_100 = trans.convertGreyscale(img_100,save)
  gray_40 = trans.convertGreyscale(img_40,save)
  canny_100 = trans.cannyEdgeDetection(gray_100,save)
  canny_40 = trans.cannyEdgeDetection(gray_40,save)
  OTSU_100 = trans.OTSUQuantize(gray_100,save)
  HLS_100 = cv2.cvtColor(img_100, cv2.COLOR_BGR2HLS)
  hue_100,lumninance_100,saturation_100 = cv2.split(HLS_100)

  # simpleBlur
  vector["simpleBlur_gif_100"] = get_ratio(compress(trans.simpleBlur(img_100,save),"gif"),baselines["gif_100"])
  vector["simpleBlur_gif_40"]  = get_ratio(compress(trans.simpleBlur(img_40,save),"gif"),baselines["gif_40"])
  vector["simpleBlur_png_100"] = get_ratio(compress(trans.simpleBlur(img_100,save),"png"),baselines["png_100"])
  vector["simpleBlur_png_40"]  = get_ratio(compress(trans.simpleBlur(img_40,save),"png"),baselines["png_40"])
  # gaussianBlur
  vector["gaussianBlur_gif_100"] = get_ratio(compress(trans.gaussianBlur(img_100,save),"gif"),baselines["gif_100"])
  vector["gaussianBlur_gif_40"] = get_ratio(compress(trans.gaussianBlur(img_40,save),"gif"),baselines["gif_40"])
  vector["gaussianBlur_png_100"] = get_ratio(compress(trans.gaussianBlur(img_100,save),"png"),baselines["png_100"])
  vector["gaussianBlur_png_40"] = get_ratio(compress(trans.gaussianBlur(img_40,save),"png"),baselines["png_40"])
  # distanceTransform
  vector["distanceTransform_gif_100"] = get_ratio(compress(trans.distanceTransform(gray_100,save),"gif"),baselines["gif_100"])
  vector["distanceTransform_gif_40"] = get_ratio(compress(trans.distanceTransform(gray_40,save),"gif"),baselines["gif_40"])
  # hardBlur
  vector["hardBlur_gif_100"] = get_ratio(compress(trans.hardBlur(img_100,save),"gif"),baselines["gif_100"])
  # convertGreyscale
  vector["grayscale_gif_100"] = get_ratio(compress(gray_100,"gif"),baselines["gif_100"])
  vector["grayscale_gif_40"] = get_ratio(compress(gray_40,"gif"),baselines["gif_40"])
  # cannyEdgeDetection
  vector["cannyEdgeDetection_gif_100"] = get_ratio(compress(canny_100,"gif"),baselines["gif_100"])
  vector["cannyEdgeDetection_gif_40"] = get_ratio(compress(canny_40,"gif"),baselines["gif_40"])
  # sobelEdgeDetection
  vector["sobelEdgeDetection_gif_100"] = get_ratio(compress(trans.sobelEdgeDetection(gray_100,save),"gif"),baselines["gif_100"])
  # laplacianDetection
  vector["laplacianDetection_gif_100"] = get_ratio(compress(trans.laplacianDetection(gray_100,save),"gif"),baselines["gif_100"])
  # sobelplusblurEdgeDetection
  vector["sobelplusblurEdgeDetection_gif_100"] = get_ratio(compress(trans.sobelplusblurEdgeDetection(gray_100,save),"gif"),baselines["gif_100"])
  # gaborKernel
  vector["gaborKernel_gif_100"] = get_ratio(compress(trans.gaborKernel(img_100,save),"gif"),baselines["gif_100"])
  # gaborGreyKernel
  vector["gaborGreyKernel_gif_100"] = get_ratio(compress(trans.gaborGreyKernel(gray_100,save),"gif"),baselines["gif_100"])
  # filter2DKernel1
  vector["filter2DKernel1_gif_100"] = get_ratio(compress(trans.filter2DKernel1(gray_100,save),"gif"),baselines["gif_100"])
  # embossFilter
  vector["embossFilter_gif_100"] = get_ratio(compress(trans.embossFilter(gray_100,save),"gif"),baselines["gif_100"])
  vector["embossFilter_gif_40"] = get_ratio(compress(trans.embossFilter(gray_40,save),"gif"),baselines["gif_40"])
  # sobelFilter
  vector["sobelFilter_gif_100"] = get_ratio(compress(trans.sobelFilter(gray_100,save),"gif"),baselines["gif_100"])
  # boxFilter
  vector["boxFilter_gif_100"] = get_ratio(compress(trans.boxFilter(img_100,save),"gif"),baselines["gif_100"])
  # arcCosine
  vector["arcCosine_gif_100"] = get_ratio(compress(trans.arcCosine(img_100,save),"gif"),baselines["gif_100"])
  vector["arcCosine_gif_40"] = get_ratio(compress(trans.arcCosine(img_40,save),"gif"),baselines["gif_40"])
  # powerTen
  vector["powerTen_gif_100"] = get_ratio(compress(trans.powerTen(img_100,save),"gif"),baselines["gif_100"])
  vector["powerTen_gif_40"] = get_ratio(compress(trans.powerTen(img_40,save),"gif"),baselines["gif_40"])
  # squareRoot
  vector["squareRoot_gif_100"] = get_ratio(compress(trans.squareRoot(img_100,save),"gif"),baselines["gif_100"])
  # brightness
  vector["brightness_gif_100"] = get_ratio(compress(trans.brightness(img_100,save),"gif"),baselines["gif_100"])
  # saturation
  vector["saturation_gif_100"] = get_ratio(compress(trans.saturation(HLS_100,save),"gif"),baselines["gif_100"])
  # pixelate10
  vector["pixelate10_Linear_gif_100"] = get_ratio(compress(trans.pixelate(img_100,10,save,method=1),"gif"),baselines["gif_100"])
  vector["pixelate10_Cubic_gif_100"] = get_ratio(compress(trans.pixelate(img_100,10,save,method=2),"gif"),baselines["gif_100"])
  vector["pixelate10_Area_gif_100"] = get_ratio(compress(trans.pixelate(img_100,10,save,method=3),"gif"),baselines["gif_100"])
  vector["pixelate10_Nearest_gif_100"] = get_ratio(compress(trans.pixelate(img_100,10,save,method=6),"gif"),baselines["gif_100"])
  # pixelate30
  vector["pixelate30_Linear_gif_100"] = get_ratio(compress(trans.pixelate(img_100,30,save,method=1),"gif"),baselines["gif_100"])
  vector["pixelate30_Cubic_gif_100"] = get_ratio(compress(trans.pixelate(img_100,30,save,method=2),"gif"),baselines["gif_100"])
  vector["pixelate30_Area_gif_100"] = get_ratio(compress(trans.pixelate(img_100,30,save,method=3),"gif"),baselines["gif_100"])
  vector["pixelate30_Nearest_gif_100"] = get_ratio(compress(trans.pixelate(img_100,30,save,method=6),"gif"),baselines["gif_100"])
  # meansDenoising5
  vector["meansDenoising5_gif_100"] = get_ratio(compress(trans.meansDenoising(img_100,5,save),"gif"),baselines["gif_100"])
  # meansDenoising30
  vector["meansDenoising30_gif_100"] = get_ratio(compress(trans.meansDenoising(img_100,30,save),"gif"),baselines["gif_100"])
  # kMeansQuantize3
  vector["kMeansQuantize3_gif_100"] = get_ratio(compress(trans.kMeansQuantize(img_100,save,nCluster=3),"gif"),baselines["gif_100"])
  vector["kMeansQuantize3_gif_40"] = get_ratio(compress(trans.kMeansQuantize(img_40,save,nCluster=3),"gif"),baselines["gif_40"])
  # kMeansQuantize12
  vector["kMeansQuantize12_gif_100"] = get_ratio(compress(trans.kMeansQuantize(img_100,save, nCluster=12),"gif"),baselines["gif_100"])
  # bwQuantizeOTSU
  vector["bwQuantizeOTSU_gif_100"] = get_ratio(compress(OTSU_100,"gif"),baselines["gif_100"])
  # HoughLines
  # This looks for lines in Canny. Could then draw the lines on white background but I don't think this kind of generative approach is useful for this method. See https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
  # bwQuantizeThreshold
  vector["bwQuantizeThreshold_gif_100"] = get_ratio(compress(trans.bwQuantizeThreshold(gray_100,save),"gif"),baselines["gif_100"])  

  blur_100 = cv2.blur(img_100,(7,7))
  blurBaseline = compress(blur_100,"gif")
  # floodFillLines
  vector["floodFillUpperThird_gif_100"] = get_ratio(compress(trans.floodFill(blur_100,"H_upperThird",save),"gif"),blurBaseline)
  vector["floodFillHorizontal_gif_100"] = get_ratio(compress(trans.floodFill(blur_100,"H_horizontal",save),"gif"),blurBaseline)
  vector["floodFillLowerThird_gif_100"] = get_ratio(compress(trans.floodFill(blur_100,"H_lowerThird",save),"gif"),blurBaseline)
  vector["floodFillLeftThird_gif_100"] = get_ratio(compress(trans.floodFill(blur_100,"V_leftThird",save),"gif"),blurBaseline)
  vector["floodFillVertical_gif_100"] = get_ratio(compress(trans.floodFill(blur_100,"V_vertical",save),"gif"),blurBaseline)
  vector["floodFillRightThird_gif_100"] = get_ratio(compress(trans.floodFill(blur_100,"V_rightThird",save),"gif"),blurBaseline)
  # floodFillPoints
  vector["floodFillMiddle_gif_100"] = get_ratio(compress(trans.floodFill(blur_100,"middle",save),"gif"),blurBaseline)
  vector["floodFillUpperLeftThird_gif_100"] = get_ratio(compress(trans.floodFill(blur_100,"upperLeftThird",save),"gif"),blurBaseline)
  vector["floodFillUpperRightThird_gif_100"] = get_ratio(compress(trans.floodFill(blur_100,"upperRightThird",save),"gif"),blurBaseline)
  vector["floodFillLowerLeftThird_gif_100"] = get_ratio(compress(trans.floodFill(blur_100,"lowerLeftThird",save),"gif"),blurBaseline)
  vector["floodFillLowerRightThird_gif_100"] = get_ratio(compress(trans.floodFill(blur_100,"lowerRightThird",save),"gif"),blurBaseline)

  # Statistics
  # fractalDimensionOTSU
  vector["fractalDimensionOTSU_100"] = stats.fractalDimension(OTSU_100/255)
  # fractalDimensionCanny
  vector["fractalDimensionCanny_100"] = stats.fractalDimension(canny_100/255)
  # fractalDimensionTreshold
  vector["fractalDimensionTreshold_100"] = stats.fractalDimension(gray_100, 0.9)
  # colorFreqDistStats
  freqStats = stats.colorFreqDistStats(hue_100)
  vector["colorFreqDistMin_100"] = freqStats["min"]
  vector["colorFreqDistMax_100"] = freqStats["max"]
  vector["colorFreqDistMean_100"] = freqStats["mean"]
  vector["colorFreqDistMedian_100"] = freqStats["median"]
  vector["colorFreqDistStd_100"] = freqStats["std"]
  vector["colorFreqDistEntropy_100"] = freqStats["entropy"]
  # colorfulnessHasler
  vector["colorfulnessHasler_100"] = stats.colorfulnessHasler(img_100)
  # luminosityRange
  vector["luminosityRange_100"] = stats.luminosityRange(lumninance_100)
  # luminosityStd
  vector["luminosityStd_100"] = stats.luminosityStd(lumninance_100)

  return vector
