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
import math
import pandas as pd
from datetime import datetime
from compression import compressComplexity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

inputFolder = "testset/"
outputFolder = "output/"
exportFolder = "image_export/"

#----------------------------------------------------------------------------------------------
# Utils
#----------------------------------------------------------------------------------------------
def resizeImage(filename):
  IMG = cv2.imread(inputFolder + filename)
  # resize to 160k pixel
  h,w,_ = IMG.shape
  if w*h > 160000:
      scale = 400/(math.sqrt(w)*math.sqrt(h))    # pixel number close to 160000
      dim = (int(w*scale),int(h*scale))
      IMG = cv2.resize(IMG, dim, interpolation=cv2.INTER_LINEAR)
  return IMG

def getTimestamp():
  dt = datetime.now()
  # getting the timestamp
  ts = datetime.timestamp(dt)
  return ts

def makePCA(df):
  embeddings = df.values
  filenames = df.index
  embeddings = StandardScaler().fit_transform(df.values)
  pca = PCA(n_components=len(df.columns))
  principalComponents = pca.fit_transform(embeddings)
  return pd.DataFrame(data = principalComponents, index=filenames)


#----------------------------------------------------------------------------------------------
# Produce transformation images for visualization, evaluation and selection
#----------------------------------------------------------------------------------------------
filename = "T01920.jpg"
testImage = resizeImage(filename)
testVector = compressComplexity(testImage,save=True)
vectorDescriptions = list(testVector.keys())

# testVector = compressComposition(testImage,save=False)