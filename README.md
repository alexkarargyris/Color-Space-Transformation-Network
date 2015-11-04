# Color Space Transformation Network
This is a color space network module that can plug into a neural network. 
It is a network layer that adjusts its configuration after training on a colored image dataset. Its output is a 3x3 transformation that is applied to the original 3-channeled images to help increase network's overall classification accuracy. 

More technical information about the architecture can be found in the following short paper: http://arxiv.org/abs/1511.01064

It uses Lasagne. Please follow instructions here: http://lasagne.readthedocs.org/en/latest/user/installation.html

- `Color_Transformation_Network.ipynb` contains a demonstration of the color network coupled with a CNN network using CIFAR-10 dataset
- `Without_Color_Transformation_Network.ipynb` constains a demonstration of the same baseline CNN network using CIFAR-10 dataset
- `colortransformationlayer.py` contains the code for the color space transformation layer. It basically multiplies the input colors R,G,B with the 3x3 output of a dense layer (i.e. fully-connected layer) 
