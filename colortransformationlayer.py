import theano
import theano.tensor as T
import lasagne
import numpy
from theano.tensor.nnet import conv

class ColorTransformationLayer(lasagne.layers.MergeLayer):
    def __init__(self, incoming, **kwargs):
        super(ColorTransformationLayer, self).__init__(incoming, **kwargs)
        conv_shp, color_shp = self.input_shapes
	if color_shp[-1] != 9:
            raise ValueError("The color network must have 9 outputs")

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
	inputImages, W = inputs
	num_batch, num_channels, height, width = inputImages.shape
	W = T.reshape(W,(-1,3,3))
	inputImages = T.reshape(inputImages,(num_batch, height, width, num_channels))
	output = T.batched_dot(inputImages, W)

	output = T.reshape(output,(num_batch, height, width, num_channels))
	output = output.dimshuffle(0, 3, 1, 2)
	return output
