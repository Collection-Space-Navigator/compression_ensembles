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
import os
import cv2
import math
import pandas as pd
from datetime import datetime
from compression import compressComplexity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
#----------------------------------------------------------------------------------------------
# Global variables - settings
#----------------------------------------------------------------------------------------------

outputFolder = "output/"
exportFolder = "image_export/"
supportformats = ('.png', '.jpg', '.jpeg')

#----------------------------------------------------------------------------------------------
# Utils
#----------------------------------------------------------------------------------------------
class compressionEmbeddings:
  def resizeImage(self, filepath):
    print('resize filepath:', filepath)
    IMG = cv2.imread(filepath)
    # resize to 160k pixel
    h,w,_ = IMG.shape
    if w*h > 160000:
        scale = 400/(math.sqrt(w)*math.sqrt(h))    # pixel number close to 160000
        dim = (int(w*scale),int(h*scale))
        IMG = cv2.resize(IMG, dim, interpolation=cv2.INTER_LINEAR)
    return IMG

  def getTimestamp(self):
    dt = datetime.now()
    # getting the timestamp
    ts = datetime.timestamp(dt)
    return ts

  def makePCA(self,df):
    embeddings = df.values
    filenames = df.index
    embeddings = StandardScaler().fit_transform(df.values)
    pca = PCA(n_components=len(df.columns))
    principalComponents = pca.fit_transform(embeddings)
    return pd.DataFrame(data = principalComponents, index=filenames)

  def processFile(self, path,filename, debug):
    if path=='':
      full_path = filename
    else:
      full_path = path+"/"+filename
    preparedImage = self.resizeImage(full_path)
    vectorAr = compressComplexity(preparedImage,save=debug)
    vectorAr['file'] = filename
    #vectorDescriptions = list(testVector.keys())
    return vectorAr

  def processFolder(self, dir_path, debug):
    res = []
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)) and path.lower().endswith(supportformats):
            res.append(path)
    json = []
    for path in res:
      json.append( self.processFile(dir_path,path,debug) )
    return json

  def process(self,path,typeFast=True,debug=False):
    data = []
    if debug:
      isExist_exportFolder = os.path.exists(exportFolder)
      if not isExist_exportFolder:
        # Create a new directory because it does not exist
        os.makedirs(exportFolder)
        print("The export directory was created!")

      isExist_outputFolder = os.path.exists(outputFolder)
      if not isExist_outputFolder:
        # Create a new directory because it does not exist
        os.makedirs(outputFolder)
        print("The output directory was created!")

    if True:
      if os.path.isdir(path):  
        print("Process this directory:",path)  
        data = self.processFolder(path,debug)
      elif os.path.isfile(path):  
        print("Process this file:",path)
        data = [self.processFile('',path,debug)]
    '''
    try:
    except:
      print("No processing file or folder defined as argument")
    '''
    # convert json to CSV
    df = pd.DataFrame(data)
    output = str(path.replace('.','_').replace('/','_')+'_embeddings.csv')
    df.to_csv(output)

#----------------------------------------------------------------------------------------------
# Produce transformation images for visualization, evaluation and selection
#----------------------------------------------------------------------------------------------

if __name__ == '__main__':
  path = sys.argv[1]
  myCompressionEmbeddings = compressionEmbeddings()
  try:
    typeFast = sys.argv[2]
    debug = sys.argv[3]
  except:
    debug = False
    typeFast = True
  myCompressionEmbeddings.process(path,typeFast,debug)