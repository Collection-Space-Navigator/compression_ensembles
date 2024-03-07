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
import sys, math

#----------------------------------------------------------------------------------------------
# Statistical transformation methods
#----------------------------------------------------------------------------------------------

def fractalDimension(Z,threshold=None):
    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])
    # Transform Z into a binary array
    if threshold:
      Z = Z/256.0
      Z = (Z < threshold)
    # Minimal dimension of image
    p = min(Z.shape)
    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))
    # Extract the exponent
    n = int(np.log(n)/np.log(2))
    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)
    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    # Fit the successive log(sizes) with log (counts)
    try:
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]
    except:
        return 0

#  Entropy of Hough Lines angles: Hough Lines are somewhat probleatic, should be reconsidered if useful
def entropy(labels, base=None):
  """ Computes entropy of label distribution. """
  n_labels = len(labels)
  if n_labels <= 1:
    return 0
  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)
  if n_classes <= 1:
    return 0
  ent = 0.
  # Compute entropy
  base = math.e if base is None else base
  for i in probs:
    ent -= i * math.log(i, base)
  return ent

def colorFreqDistStats(H):
  h, w = H.shape
  tot = h * w
  # calculate histogram
  hist = cv2.calcHist([H], [0], None, [181], [0, 181])
  # Normalize mean
  nmean = np.mean(hist)/180
  Hmin = hist.min()/180
  Hmax = hist.max()/180
  nmed = np.median(hist)/180
  nstd = np.std(hist)/180
  ent = entropy(hist)
  # # entropy = -sum(hist[i]/tot)*log(hist[i]/tot))
  # # where hist[i]/tot = probability
  # # note: log(0) = -inf, so skip empty bins
  # entropy2 = 0
  # bins = 0
  # for i in range (0,181):
  #     if hist[i][0] != 0:
  #         entropy2 = entropy2 - (hist[i][0]/tot)*math.log(hist[i][0]/tot)
  #         bins = bins + 1
  # print('entropy2:', entropy2)
  # print('entropy:', ent)
  return {"mean": nmean, "min": Hmin, "max": Hmax, "median": nmed, "std": nstd, "entropy": ent}

# Hasler & Süsstrunk -> hasler2003measuring: https://doi.org/10.1117/12.477378.
def colorfulnessHasler(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)

def luminosityRange(L):
  return abs(L.max()-L.min())

def luminosityStd(L):
  return L.std()
