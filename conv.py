"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max. 
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
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

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
import matplotlib.pyplot as plt
from test import test


def relu(x):
    return theano.tensor.switch(x<0, 0, x)

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]



def evaluate_lenet5(learning_rate=0.1,n_epochs=500,
                    dataset='cifar10',
                    nkerns=[32,32],batch_size=500):


	
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    #nkerns=[0,0]
    titre=raw_input('Id de la simulation  :\n')
    dep=raw_input('premiere couche a etre binarisee (default : 1)\n')
    nkerns[0]=raw_input('dimension de la premiere couche (default :32)\n')
    nkerns[1]=raw_input('dimension de la deuxieme couche (default : 32)\n')
    learning_rate=raw_input('learning rate (default : 0.01)\n')
    alpha_rate=raw_input('alpha rate (default 1.1)\n')

    if titre=='':
        titre='current'
    if dep=='':
        dep=-1
    if nkerns[0]=='':
        nkerns[0]=32
    if nkerns[1]=='':
        nkerns[1]=32
    if learning_rate=='':
        learning_rate=0.01
    if alpha_rate=='':
        alpha_rate=1.1

    dep=int(dep)
    nkerns[0]=int(nkerns[0])
    nkerns[1]=int(nkerns[1])
    learning_rate=float(learning_rate)
    alpha_rate=float(alpha_rate)


    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    #print train_set_x[0,:].eval()
    #print train_set_x.shape.eval()  #50000*784
    #print train_set_y.shape.eval()  #50000
    #print valid_set_x.shape.eval()  #10000*784
    #print valid_set_y.shape.eval()  #10000
    #print test_set_x.shape.eval()   #10000*784
    #print test_set_y.shape.eval()   #10000

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
                         

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3,32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(2, 2)
    )
    #size = nkerns[0]*5*5 = 500

    #outputLayer0=layer0.output
    #array=outputLayer0.eval({x: test_set_x[0:500, :].eval()})
    #array1=array[10, 0, :, :]
    #array2=array[20, 0, :, :]
    #arr=[array1,array2]
    #pylab.imshow(arr[1], cmap=pylab.gray())

    #f = pylab.figure()
    #for i in range(1, 2):
    #    f.add_subplot(1, 2, i)  # this line outputs images on top of each other
    #    pylab.imshow(arr[i-1],cmap=pylab.gray())
    #pylab.title('Double image')
    #pylab.show()
    
   

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 14, 14),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )



    #size = nkerns[1]*nkerns[0]*5*5 = 500*50 = 25000


    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 5 * 5,
        n_out=500,
        activation=relu
    )

    #size=n_in*n_out (def is 400 000)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    #size = n_in*n-out (def is 5000)
    #layer3_binary = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    #layer3_binary.W=1*(layer3_binary.W>0.5)
    
    # the cost we minimize during training is the NLL of the model
    #cost = layer3.negative_log_likelihood(y)

    alpha=1;
    cl0=0
    cl1=0
    cl2=0
    cl3=0
    
    if dep<=3:
        cl3=(((layer3.W+0.01)**2)*(layer3.W-0.01)**2).sum()
    if dep<=2:
        cl2=(((layer2.W+0.01)**2)*(layer2.W-0.01)**2).sum()
    if dep<=1:
        cl1=(((layer1.W+0.01)**2)*(layer1.W-0.01)**2).sum()
    if dep==0:
        cl0=(((layer0.W+0.01)**2)*(layer0.W-0.01)**2).sum()

    # if dep==3:
    #     cl3=(((layer3.W+0.01)**2)*(layer3.W-0.01)**2).sum()
    # if dep>=2:
    #     cl2=(((layer2.W+0.01)**2)*(layer2.W-0.01)**2).sum()
    # if dep>=1:
    #     cl1=(((layer1.W+0.01)**2)*(layer1.W-0.01)**2).sum()
    # if dep>=0:
    #     cl0=(((layer0.W+0.01)**2)*(layer0.W-0.01)**2).sum()

    #if dep<=3:
    #    cl3=(((layer3.W)**2)*(layer3.W-0.02)**2).sum()
    #if dep<=2:
    #    cl2=(((layer2.W)**2)*(layer2.W-0.02)**2).sum()
    #if dep<=1:
    #    cl1=(((layer1.W)**2)*(layer1.W-0.02)**2).sum()
    #if dep==0:
    #    cl0=(((layer0.W)**2)*(layer0.W-0.02)**2).sum()

    cost=(layer3.negative_log_likelihood(y)+alpha*(cl0+cl1+cl2+cl3))

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    ##grads=params;


    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='ignore',

    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'


    

    fichier = open(titre+"-p","a")
    fichier.write('---- New Train ---- \n' %
                ())

    fichier.close()


    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    
    epoch = 0
    done_looping = False
    
    """titreCharge='TAM35sauv2300-p'
    f = file(titreCharge, 'rb')
    params = cPickle.load(f)
    f.close()
    layer0.W=params[6]
    layer1.W=params[4]
    layer2.W=params[2]
    layer1.W=params[0]

    layer0.b=params[7]
    layer1.b=params[5]
    layer2.b=params[3]
    layer1.b=params[1]"""



    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            
            #pylab.hist(layer3.W.get_value(), bins=50)
            #pylab.show()
            

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            print titre,dep,' - ',cost_ij
            
            #print sum(sum((layer3.W.get_value()**2)*((layer3.W.get_value()-1)**2)))

            if (iter + 1) % 100== 0:
                # plt.hist(layer3.W.get_value(), 50, normed=1, facecolor='g', alpha=0.75)
                # plt.show()
                # compute zero-one loss on validation set
                f = file(titre+'sauv'+str(iter+1)+'-p', 'wb')
                cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()
                this_binary_validation_loss=test(titre+'sauv'+str(iter+1)+'-p',dep)

                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f, binary validation error %s %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.,
                       this_binary_validation_loss*100))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%  - binary test error of bet model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.,this_binary_validation_loss*100))
                learning_rate=learning_rate/1.1
            	if this_validation_loss < 1.1*best_validation_loss:
            		alpha=alpha_rate*alpha
                        alpha_status = 'inc'
                        print (('Alpha augmente --> %f') % alpha)
            	else:
            		alpha=1/alpha_rate*alpha
            		print (('Alpha diminue --> %f') % alpha)
                        alpha_status='dec'


                fichier = open(titre+'-p',"a")
                fichier.write('%s - %i - epoch %i - minibatch %i/%i - c : %2.4f - a %5.0f %s - LR : %1.3f - nkerns %3.0f - %3.0f - t %1.0f - '
                            't %2.2f %%  - bt  %2.2f %% \n' %
                            (titre,dep,epoch, minibatch_index + 1, n_train_batches,this_validation_loss,alpha,alpha_status,learning_rate,nkerns[0],
                            nkerns[1],(time.clock()-start_time)/60,test_score*100.,this_binary_validation_loss*100))

                fichier.close()

                



                cost=(layer3.negative_log_likelihood(y)+alpha*(cl0+cl1+cl2+cl3))

                train_model = theano.function(
                        [index],
				        cost,
				        updates=updates,
				        givens={
				            x: train_set_x[index * batch_size: (index + 1) * batch_size],
				            y: train_set_y[index * batch_size: (index + 1) * batch_size]
				        },
				        on_unused_input='ignore',
                    )





            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
