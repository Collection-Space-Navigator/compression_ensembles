# Compression Ensembles for Python

### Alternative Python implementation for compression ensembles to quantify the aesthetic complexity of images

See paper: https://epjds.epj.org/articles/epjdata/abs/2023/01/13688_2023_Article_397/13688_2023_Article_397.html

"Compression ensembles quantify aesthetic complexity and the evolution of visual art" (2023)
Andres Karjus, Mar Canet Solà, Tillmann Ohm, Sebastian E. Ahnert, Maximilian Schich

Our paper may describe slightly different transformations using R and ImageMagick: 
https://github.com/andreskarjus/compression_ensembles 
Use R code and data to replicate the analyses in "Compression ensembles quantify aesthetic complexity and the evolution of visual art" by Karjus et al 2023.

The Python version presented in this repository uses OpenCV and PIL with optimized transformations that significantly reduce computation time. 
The specific transformations and total number are arbitrary for the method as stated in the supplementary material:

>"Note that the compression ensemble approach is general and the specific number of transformations is unimportant. As demonstrated in the Evaluation section in the main text of the paper, and handful transformations may be enough for some tasks. Here, we opted to implement a fairly large array of transformations and additional compressions with different sizes and algorithms, in order to explore the transformation space in a relatively comprehensive manner. If, unlike here, computation time is important, a smaller set of transformations is likely a better choice."

## Acknowledgements
AK, MCS, TO, MS are supported by ERA Chair for <a href="https://cudan.tlu.ee/" target="_blank">Cultural Data Analytics</a>, funded through the European Union’s Horizon 2020 research and innovation program (Grant No.810961).
