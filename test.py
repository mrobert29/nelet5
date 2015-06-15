import os
import sys
import time

import numpy
import pylab

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import cPickle

from logistic_sgd import  load_data
import matplotlib.pyplot as plt


class LeNetConvPoolLayer(object):
    def __init__(self, W,b,rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        #self.W = 0.01*(W>0)-0.01*(W<0)
        self.W=W
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = b
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]


class LogisticRegression(object):
    def __init__(self,W,b, input, n_in, n_out):
        # self.W = 0.01*(W>0)-0.01*(W<0)
        self.W=W
        self.b = b
        self.L1 = (
            abs(self.W).sum()
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        
   	def negative_log_likelihood(self, y):
   		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self,W,b, rng, input, n_in, n_out, 
                 activation=T.tanh):
    	self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        # self.W = 0.01*(W>0)-0.01*(W<0)
        self.W=W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

###---TEST---###

def test(titre,dep):
	print 'debut du test de binarisation - a partir de la couche ',dep
	rng = numpy.random.RandomState(23455)
	batch_size=500
	n_epochs=200
	nkerns=[20, 50]


	f = file(titre, 'rb')
	params = cPickle.load(f)
	f.close()


	if dep<=3:
		params[0]=0.01*(params[0]>0)-0.01*(params[0]<0)	
	if dep<=2:
		params[2]=0.01*(params[2]>0)-0.01*(params[2]<0)
	if dep<=1:
		params[4]=0.01*(params[4]>0)-0.01*(params[4]<0)
	if dep==0:
		params[6]=0.01*(params[6]>0)-0.01*(params[6]<0)	




	dataset='mnist.pkl.gz'
	datasets = load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]
	n_train_batches /= batch_size
	n_valid_batches /= batch_size
	n_test_batches /= batch_size
	index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')   # the data is presented as rasterized images
	y = T.ivector('y')  # the lab


	print '... building model'

	layer0_input = x.reshape((batch_size, 1, 28, 28))

	layer0 = LeNetConvPoolLayer(
	    params[6],
	    params[7],
	    rng,
	    input=layer0_input,
	    image_shape=(batch_size, 1, 28, 28),
	    filter_shape=(nkerns[0], 1, 5, 5),
	    poolsize=(2, 2)
	)


	layer1 = LeNetConvPoolLayer(
	    params[4],
	    params[5],
	    rng,
	    input=layer0.output,
	    image_shape=(batch_size, nkerns[0], 12, 12),
	    filter_shape=(nkerns[1], nkerns[0], 5, 5),
	    poolsize=(2, 2)
	)

	layer2_input = layer1.output.flatten(2)

	layer2=HiddenLayer(
		params[2],
		params[3],
		rng,
		input=layer2_input,
		n_in=nkerns[1] * 4 * 4,
		n_out=500,
		activation=T.tanh
	)

	layer3=LogisticRegression(params[0],params[1],input=layer2.output, n_in=500, n_out=10)


	print '... testing'
	test_model = theano.function(
	    [index],
	    layer3.errors(y),
	    givens={
	        x: test_set_x[index * batch_size: (index + 1) * batch_size],
	        y: test_set_y[index * batch_size: (index + 1) * batch_size]
	    }
	)

	test_losses = [
	    test_model(i)
	    for i in xrange(n_test_batches)
	]
	test_score = numpy.mean(test_losses)

	print 'binary test achieve - ',test_score
	return test_score	



if __name__ == '__main__':
	titre=raw_input('Nom du fichier a analyser  :\n')
	dep=input('premiere couche a etre binarisee :\n')
	test(titre,dep)