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
import shutil
import concurrent.futures

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
    # print('resize filepath:', filepath)
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
    df = df.dropna(axis=1)
    embeddings = df.values
    filenames = df.index
    embeddings = StandardScaler().fit_transform(df.values)
    pca = PCA(n_components=min(len(df.columns),len(df.index)))
    principalComponents = pca.fit_transform(embeddings)
    return pd.DataFrame(data = principalComponents, index=filenames)

  def processFile(self, path,filename, debug):
    if path=='':
      full_path = filename
    else:
      full_path = path+"/"+filename
    preparedImage = self.resizeImage(full_path)
    # make greyscale
    preparedImage = cv2.cvtColor(preparedImage, cv2.COLOR_BGR2GRAY)
    # convert back to rgb
    preparedImage = cv2.cvtColor(preparedImage, cv2.COLOR_GRAY2RGB)
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
  
  def delete_batches(self):
    # Delete all files in batches directory
    folder = 'batches'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
      
  def process(self, path, typeFast=True, debug=False, pca=True):
    batch_size = 50
    batch_count = 0
    batch_data = []

    def process_single_file(file):
        return self.processFile(path, file, debug)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(supportformats)]
        results = list(executor.map(process_single_file, files))

    for result in results:
        if len(batch_data) < batch_size:
            batch_data.append(result)
        else:
            batch_count += 1
            df = pd.DataFrame(batch_data)
            df.set_index("file", inplace=True)
            output = f'batches/{batch_count}.csv'
            df.to_csv(output)
            batch_data = []

    if batch_data:
        batch_count += 1
        df = pd.DataFrame(batch_data)
        df.set_index("file", inplace=True)
        output = f'batches/{batch_count}.csv'
        df.to_csv(output)

    self.makeEnsemble(path, pca)

  def makeEnsemble(self, path, pca):
    try:
      all_files_data = []
      for file in os.listdir("batches"):
          if file.endswith(".csv"):
              df = pd.read_csv(os.path.join("batches", file), index_col="file")
              all_files_data.append(df)
      concatenated_data = pd.concat(all_files_data)
      name = str(path.replace('.', '_').replace('/', '_'))
      concatenated_data.to_csv(f"output/{name}_raw_ensembles.csv")
    except:
      print("No batches found")
    else:
      self.delete_batches()
      if pca:
        df = self.makePCA(concatenated_data)
        df.to_csv(f"output/{name}_pca_ensembles.csv")



if __name__ == '__main__':
  myCompressionEmbeddings = compressionEmbeddings()
  path = "input"
  typeFast = True
  debug = True
  myCompressionEmbeddings.process(path, typeFast, debug)

