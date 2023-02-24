### Python implementation for compression ensembles to quantify the aesthetic complexity of images

See paper: https://arxiv.org/abs/2205.10271
"Compression ensembles quantify aesthetic complexity and the evolution of visual art"
Andres Karjus, Mar Canet Solà, Tillmann Ohm, Sebastian E. Ahnert, Maximilian Schich

Our paper may describe slightly different transformations using R and ImageMagick.  
The Python version presented in this repository uses OpenCV and PIL with optimized transformations that should significantly reduce computation time. 
The specific transformations and total number are arbitrary for the method as stated in the supplementary material:

>"Note that the compression ensemble approach is general and the specific number of transformations is unimportant. As demonstrated in the Evaluation section in the main text of the paper, and handful transformations may be enough for some tasks. Here, we opted to implement a fairly large array of transformations and additional compressions with different sizes and algorithms, in order to explore the transformation space in a relatively comprehensive manner. If, unlike here, computation time is important, a smaller set of transformations is likely a better choice."