

import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample

from logistic_sgd import LogisticRegression, load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

import matplotlib.pyplot as plt


# start-snippet-1
class WTA_bis(object):
    


    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
	n_specific=None,
	n_common=None,
        W_prime=None,
        bvis=None,
        block_size=28,
        histogram=None,
        cliques=None,
        
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
	self.n_specific = n_specific
        self.n_hidden = n_hidden
        n_common = n_hidden - 10 * n_specific
        self.n_common = n_common
        self.block_size=block_size

        if not histogram:
            n_block=784/block_size
            #initial_H=np.ndarray(shape=(10,n_block, block_size), dtype=np.int32)
            initial_H=np.zeros(shape=(10,n_block, block_size), dtype=np.int32)
   

        histogram = theano.shared(value=initial_H, name='histogram', borrow=True)
        self.histogram=histogram

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        initial_specW = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_specific + n_visible)),
                    #low=0,
                    high=4 * np.sqrt(6. / (n_specific + n_visible)),
                    size=(10,n_visible, n_specific)
                ),
                dtype=theano.config.floatX
            )
        specW = theano.shared(value=initial_specW, borrow=True)

        initial_commonW = np.asarray(
            numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (n_common + n_visible)),
                #low=0,
                high=4 * np.sqrt(6. / (n_common + n_visible)),
                size=(n_visible, n_common)
            ),
            dtype=theano.config.floatX
        )
	commonW = theano.shared(value=initial_commonW, borrow=True)

        # non tied weights
            
        initial_specW_prime = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_specific + n_visible)),
                    #low=0,
                    high=4 * np.sqrt(6. / (n_specific + n_visible)),
                    size=(10, n_specific, n_visible)
                ),
                dtype=theano.config.floatX
            )
	specW_prime = theano.shared(value=initial_specW_prime, borrow=True)

        initial_commonW_prime = np.asarray(
            numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (n_common + n_visible)),
                #low=0,
                high=4 * np.sqrt(6. / (n_common + n_visible)),
                size=(n_common, n_visible)
            ),
            dtype=theano.config.floatX
        )
	commonW_prime = theano.shared(value=initial_commonW_prime, borrow=True)

        initialW_clique = np.asarray(
            numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                #low=0,
                high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                size=(n_hidden, n_visible)
            ),
            dtype=theano.config.floatX
        )
	W_clique = theano.shared(value=initialW_clique, borrow=True)

        specb = theano.shared(
                value=np.zeros((10,n_specific),
                    dtype=theano.config.floatX
                ),
                borrow=True)

	comb = theano.shared(
                value=np.zeros(
                    n_common,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )	

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

	b_clique = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )	 

        if not cliques:
            n_block=n_visible // block_size
            liste = np.zeros(shape=(10, n_block))
            for i in xrange(10):
                for j in xrange(n_block):
                    liste[i][j]=numpy_rng.randint(0, block_size)
            
            clique_values=np.zeros(
                    shape=(10, n_block, block_size),
                    dtype=theano.config.floatX
                )
            
            for i in xrange(10):
                for j in xrange(n_block):
                    for k in xrange(block_size):
                        if k == liste[i][j]:
                            clique_values[i][j][k]=1.

            cliques = theano.shared(value=clique_values, name='cliques', borrow=True)
                        
        self.cliques=cliques
        

        self.specW = specW
	self.commonW = commonW
        # b corresponds to the bias of the hidden
        self.specb = specb
	self.comb = comb
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        #self.W_prime = self.W.T
        # untied weights
        self.specW_prime=specW_prime
        self.commonW_prime=commonW_prime
	self.W_clique = W_clique
	self.b_clique = b_clique
        
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='inputx')
            self.y = T.dmatrix(name='inputy')
        else:
            self.x = input[0]
            self.y = input[1]


        self.params = [self.specW, self.commonW, self.specW_prime, self.commonW_prime, self.specb, self.comb, self.b_prime, self.W_clique, self.b_clique]
        #self.params = [self.W, self.b, self.b_prime]
    # end-snippet-1

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input, label):
        """ Computes the values of the hidden layer """
	comPart = T.dot(input, self.commonW) + self.comb
	specPart = T.dot(input, self.specW[label]) + self.specb[label]
        return (T.nnet.sigmoid(comPart),T.nnet.sigmoid(specPart))

    def get_reconstructed_input(self, com_hidden, spec_hidden, label):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        r=T.nnet.sigmoid(T.dot(com_hidden, self.commonW_prime) + T.dot(spec_hidden, self.specW_prime[label]) + self.b_prime)
        return r

    def get_test_hidden_values(self, input):
        """ Computes the values of the hidden layer """
	comPart = T.nnet.sigmoid(T.dot(input, self.commonW) + self.comb)
	specPart = theano.shared(value=np.zeros((10,1,self.n_specific)),borrow=True)
	sumScores = theano.shared(value=np.zeros(10),borrow=True)
	ActnCount = theano.shared(value=np.zeros(10),borrow=True)
	U = 0.5*T.ones_like(self.specb[0])
	for i in xrange(10):
            S = T.nnet.sigmoid(T.dot(input, self.specW[i]) + self.specb[i])
	    specPart = T.set_subtensor(specPart[i],S)
	    sumScores = T.set_subtensor(sumScores[i],T.sum(S))#,axis=[0,1,2])
	    ActnCount = T.set_subtensor(ActnCount[i],T.sum(S > U))#,axis=[0,1,2])'''

        '''print S.eval().shape
        print S.sum().eval().shape'''
	sumScoreWinner = T.argmax(sumScores)
	ActnCountWinner = T.argmax(ActnCount)
        return (comPart,specPart,sumScoreWinner,ActnCountWinner)

    def get_test_reconstructed_input(self, com_hidden, spec_hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
	'''r = T.dot(spec_hidden[0], self.specW_prime[0])
	for i in xrange(1,10):
	    r += T.dot(spec_hidden[i], self.specW_prime[i])
	r += T.dot(com_hidden, self.commonW_prime)''' # MODIF : Test sans les features communes
	'''hid_vect = theano.shared(value=np.zeros((1,self.n_hidden)),borrow=True)
	hid_vect = T.set_subtensor(hid_vect[0:250],com_hidden) # TODO : rendre adaptable
	for i in xrange(10):
	    hid_vect = T.set_subtensor(hid_vect[250+i*25:275+i*25], spec_hidden[i])'''
	n_common = self.n_common
	n_specific = self.n_specific
	OutAct = T.dot(spec_hidden[0], self.W_clique[n_common:n_common+n_specific])
	for i in xrange(1,10):
	    OutAct += T.dot(spec_hidden[i], self.W_clique[n_common+i*n_specific:n_common+(i+1)*n_specific])
	# OutAct += T.dot(com_hidden, self.W_clique[0:250])
	r = T.nnet.sigmoid(OutAct+self.b_clique)
        return r

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        h = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(h)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
	gsw = T.grad(cost, self.specW[self.y])
	gcw = T.grad(cost, self.commonW)
        gwprime = T.grad(cost, self.W_prime)
        gb = T.grad(cost, self.comb)
        gbprime = T.grad(cost, self.b_prime)
        # generate the list of updates
        sWupdates = [
            (sw, sw - learning_rate * g)
            for sw, g in zip([self.specW[self.y]], [gsw])
        ]
	cWupdates = [
            (cw, cw - learning_rate * g)
            for cw, g in zip([self.commonW], [gcw])
        ]
        WPupdates = [
            (wp, wp - learning_rate * gp)
            for wp, gp in zip([self.W_prime], [gwprime])
        ]
        sBupdates = [
            (sb, sb - learning_rate * g)
            for sb, g in zip([self.specb], [gsb])
        ]
	cBupdates = [
            (cb, cb - learning_rate * g)
            for cb, g in zip([self.comb], [gcb])
        ]
        BPupdates = [
            (bp, bp - learning_rate * gbp)
            for bp, gbp in zip([self.b_prime], [gbprime])
        ]
        updates = sWupdates+cWupdates+WPupdates+sBupdates+cBupdates+BPupdates

        return (cost, updates)

    def get_hidden_output(self, corruption_level, input):
        tilde_x = self.get_corrupted_input(input, corruption_level)
        hc,hs = self.get_hidden_values(tilde_x,self.y)
        return hc,hs

    #def binary_lwta(self, data, block_size):
    def binary_lwta(self, data):
        num_batches = data.shape[0]
        num_nodes = data.shape[1]
        num_blocks = num_nodes // self.block_size
        w = data.reshape((num_batches, num_blocks, self.block_size))
        block_max = w.max(axis=2).dimshuffle(0, 1, 'x') * T.ones_like(w)
        max_mask = T.cast(w >= block_max, 'float32')
        max_mask=max_mask.reshape((num_batches, num_blocks*self.block_size))

        return max_mask
        '''
        #break the tie
        indices = numpy.array(range(1, self.block_size+1))
        max_mask2 = max_mask * indices
        block_max2 = max_mask2.max(axis=2).dimshuffle(0, 1, 'x') * T.ones_like(w)
        max_mask3 = T.cast(max_mask2 >= block_max2, 'float32')
        w2 = w * max_mask3
        w3 = w2.reshape((p.shape[0], p.shape[1]))
        return w3'''

        
    def get_cost_updates_wta(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        h = self.get_hidden_values(tilde_x)
        r = self.get_reconstructed_input(h)
        z = self.binary_lwta(r)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        
        L =  T.sum((z-r)**2, axis=1)
        
        #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

    # calculate the reconstruction cost function using common and class-specific features
    # labels: y
    def get_cost_updates_ComSpecAE(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        ch,sh = self.get_hidden_values(tilde_x, self.y[0])
        r = self.get_reconstructed_input(ch,sh, self.y[0])       
        
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        
        L =  T.sum((self.x-r)**2, axis=1)

        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
	gsw = T.grad(cost, self.specW)#[self.y[0]])
	gcw = T.grad(cost, self.commonW)
        gswprime = T.grad(cost, self.specW_prime)
	gcwprime = T.grad(cost, self.commonW_prime)
        gsb = T.grad(cost, self.specb)
	gcb = T.grad(cost, self.comb)
        gbprime = T.grad(cost, self.b_prime)
        # generate the list of updates
        sWupdates = [
            (sw, sw - learning_rate * g)
            for sw, g in zip([self.specW], [gsw])#[self.y]
        ]
	cWupdates = [
            (cw, cw - learning_rate * g)
            for cw, g in zip([self.commonW], [gcw])
        ]
        sWPupdates = [
            (wp, wp - learning_rate * gp)
            for wp, gp in zip([self.specW_prime], [gswprime])
        ]
	cWPupdates = [
            (wp, wp - learning_rate * gp)
            for wp, gp in zip([self.commonW_prime], [gcwprime])
        ]
        sBupdates = [
            (sb, sb - learning_rate * g)
            for sb, g in zip([self.specb], [gsb])
        ]
	cBupdates = [
            (cb, cb - learning_rate * g)
            for cb, g in zip([self.comb], [gcb])
        ]
        BPupdates = [
            (bp, bp - learning_rate * gbp)
            for bp, gbp in zip([self.b_prime], [gbprime])
        ]
        updates = sWupdates+cWupdates+sWPupdates+cWPupdates+sBupdates+cBupdates+BPupdates

        return (cost, updates, ch)



    # calculate the cost function referring a fixed clique for each digit
    # labels: y
    def get_cost_updates_clique(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        ch,sh,ssw,acw = self.get_test_hidden_values(tilde_x)
        r = self.get_test_reconstructed_input(ch,sh)

        #z=T.zeros_like(r)
        z = theano.shared(
                value=np.zeros(
                    (1, 784),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        
        cliques=self.cliques.reshape((10, 784))
        
        for idx in xrange (1):
            z=T.set_subtensor(z[idx], cliques[self.y[idx]])
            
        
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        
        L =  T.sum((z-r)**2, axis=1)

        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
	'''gsw = T.grad(cost, self.specW)#[self.y[0]])
	gcw = T.grad(cost, self.commonW)'''# OPTION : on ne touche plus aux features dans cette phase
        gwclique = T.grad(cost, self.W_clique)
        '''gsb = T.grad(cost, self.specb)
	gcb = T.grad(cost, self.comb)'''
        gbclique = T.grad(cost, self.b_clique)
        # generate the list of updates
        WCupdates = [
            (wc, wc - learning_rate * gwc)
            for wc, gwc in zip([self.W_clique], [gwclique])
        ]
	BCupdates = [
            (bc, bc - learning_rate * gbc)
            for bc, gbc in zip([self.b_clique], [gbclique])
        ]

        updates = WCupdates+BCupdates

        return (cost, updates, ch)


    #one by one
    def estimate_clique_label(self, corruption_level):
        
        comHid, specHid, sumScoreWinner, ActnCountWinner = self.get_test_hidden_values(input=self.x)
        r_output = self.get_test_reconstructed_input(comHid, specHid)
        wta_output = self.binary_lwta(r_output)
        wta_reshape = wta_output.reshape(((1, 784//self.block_size, self.block_size)))
        wta_copy = wta_reshape * T.ones_like(self.cliques)
        scores = (wta_copy * self.cliques).sum(axis=2).sum(axis=1)
        max_index = T.argmax(scores)

        return (max_index, sumScoreWinner, ActnCountWinner)

        


    #one by one update
##    def update_histogram(self, corruption_level):
##        tilde_x = self.get_corrupted_input(self.x, corruption_level=0)
##        h = self.get_hidden_values(tilde_x)
##        r = self.get_reconstructed_input(h)
##        z = self.binary_lwta(r)
##        num_batches = z.shape[0]
##        num_nodes = z.shape[1]
##        z_reshape=z.reshape((1, num_nodes//self.block_size, self.block_size))
##                
##        histogram_updates = (self.histogram, T.set_subtensor(self.histogram[self.y], self.histogram[self.y]+z_reshape[0]))
##        return histogram_updates
##
##    
##    def estimate_label(self, corruption_level):
##        
##        tilde_x = self.get_corrupted_input(self.x, corruption_level=0)
##        h = self.get_hidden_values(tilde_x)
##        r = self.get_reconstructed_input(h)
##        z = self.binary_lwta(r)
##        num_batches = z.shape[0]
##        num_nodes = z.shape[1]
##        z_reshape=z.reshape((1, num_nodes//self.block_size, self.block_size))
##        z_copy=z_reshape * T.ones_like(self.histogram)
##        scores=(z_copy * self.histogram).sum(axis=2).sum(axis=1)
##        max_index=T.argmax(scores)
##
##        return max_index
##



def test_dA(Nspec,learning_rate=0.1, training_epochs=50,#30,#100,
            dataset='mnist.pkl.gz',
            batch_size=1, output_folder='dA_plots'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    print train_set_x.get_value(borrow=True).shape
    print valid_set_x.get_value(borrow=True).shape
    print test_set_x.get_value(borrow=True).shape

    train_set_y_values=train_set_y.eval()
    print train_set_y_values
    array=np.zeros((10,), dtype=np.int)
    for n in xrange(50000):
        array[train_set_y_values[n]]+=1
    print array
    

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    idx=T.lscalar()
    idx2=T.lscalar()
 
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = WTA_bis(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=[x, y],
        n_visible=28 * 28,
	n_specific = Nspec,
    )

    
##    cost, updates = da.get_cost_updates(
##        corruption_level=0.,
##        learning_rate=learning_rate
##    )



##    cost, updates = da.get_cost_updates_wta(
##        corruption_level=0.,
##        learning_rate=learning_rate,
##    )
##
##
##
##    train_da = theano.function(
##        [index],
##        cost,
##        updates=updates,
##        givens={
##            x: train_set_x[index * batch_size: (index + 1) * batch_size]
##        }
##    )
                      

    cost, updates, ch = da.get_cost_updates_ComSpecAE(0.3, learning_rate)

    pre_train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(50):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            #c.append(train_da(batch_index))
            c.append(pre_train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)



    cost, updates, ch = da.get_cost_updates_clique(0.3, learning_rate)


    train_clique_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )

    get_com_hid = theano.function(
        [index],
        ch,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
        }
    )    

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(50):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            #c.append(train_da(batch_index))
            c.append(train_clique_da(batch_index)) 
	    '''com_hid = get_com_hid(batch_index)
	    print "com_hid_shape : ", com_hid.shape'''

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))


##    ###############################
##    # Calculating Histogram#
##    ###############################
##
####    tilde_x = da.get_corrupted_input(x, 0)
####    h = da.get_hidden_values(tilde_x)
####    r = da.get_reconstructed_input(h)
####    z = da.binary_lwta(r)
####    num_batches = z.shape[0]
####    num_nodes = z.shape[1]
####    z_reshape=z.reshape((1, num_nodes//28, 28))
####
####    wta_out = theano.function(
####        inputs=[idx],
####        outputs=z_reshape,
####        givens={
####            x: train_set_x[idx: (idx + 1)]
####        }
####    )
####    n_train=train_set_x.get_value(borrow=True).shape[0]
####
####    for idx in xrange(n_train):
####        if train_set_y.eval()[idx]==9:
####            output=wta_out(idx)
####            print ('number %i ' % idx)
####            print output[0]
##       
##
##    histogram_updates=da.update_histogram(0)
##
##    train_histogram = theano.function(
##        inputs=[idx],
##        updates=[histogram_updates],
##        givens={
##            x: train_set_x[idx : (idx+1)],
##            y: train_set_y[idx : (idx+1)]
##        }
##    )
##    n_train=train_set_x.get_value(borrow=True).shape[0]
##    print n_train
##    print '... calculating the histogram'
##    for idx in xrange(n_train):
##        train_histogram(idx)
##
####    for n in xrange(10):
####        print('number %i ' % n)
####        print da.histogram.eval()[n]
##
##
##    
##    
##    ###########################################
##    # Estimate the test labeles via histogram #
##    ###########################################
##
####    tilde_x = da.get_corrupted_input(x, corruption_level=0)
####    h = da.get_hidden_values(tilde_x)
####    r = da.get_reconstructed_input(h)
####    z = da.binary_lwta(r)
####    num_batches = z.shape[0]
####    num_nodes = z.shape[1]
####    z_reshape=z.reshape((1, num_nodes//28, 28))
####    z_copy=z_reshape * T.ones_like(da.histogram)
####    scores=(z_copy * da.histogram).sum(axis=2).sum(axis=1)
####    #max_index=T.argmax(scores)
####
####
####    test_label_bis = theano.function(
####        inputs=[idx2],
####        outputs=z_reshape,
####        givens={
####            x: test_set_x[idx2:(idx2+1)]
####        }
####    )
####
####
####    data0=test_label_bis(0)
####    print data0
####    data1=test_label_bis(30)
####    print data1
####    data2=test_label_bis(100)
####    print data2
##
##
##    label=da.estimate_label(0)
##
##    test_label = theano.function(
##        inputs=[idx2],
##        outputs=label,
##        givens={
##            x: test_set_x[idx2:(idx2+1)]
##        }
##    )
##    
##    n_test=test_set_x.get_value(borrow=True).shape[0]
##    print n_test
##    print '... estimating the label'
##    labels=[]
##    for idx2 in xrange(n_test):
##        max_index=test_label(idx2)
##        labels.append(max_index)
##        #print max_index
##
##
##


    #################################################
    # Estimate the test labeles via WTA and cliques #
    #################################################
    label, sumScoresWinner, ActnCountWinner = da.estimate_clique_label(0)

    test_label = theano.function(
        inputs=[idx2],
        outputs=[label,sumScoresWinner,ActnCountWinner],
        givens={
            x: test_set_x[idx2:(idx2+1)]
        }
    )
    
    n_test=test_set_x.get_value(borrow=True).shape[0]
    print n_test
    print '... estimating the label'
    labels=np.zeros(n_test)
    labels2=np.zeros(n_test)
    labels3=np.zeros(n_test)
    for idx2 in xrange(n_test):
        classifs=test_label(idx2)
        labels[idx2] = classifs[0]
        labels2[idx2] = classifs[1]
        labels3[idx2] = classifs[2]
        
    test_set_y_values = test_set_y.eval()

    iden = T.cast(test_set_y_values == labels, 'int32')
    iden_values = iden.owner.inputs[0].value
    err_rate = 1 - sum(iden_values)/ float(n_test)

    print err_rate

    print 'test error rate %f %%' % (err_rate * 100.)
    f = open('ComSpecBatch.txt', 'a')
    f.write('spec = '+str(Nspec)+' : ')
    f.write(str(err_rate * 100.))
    f.write('\n')

    iden = T.cast(test_set_y_values == labels2, 'int32')
    iden_values = iden.owner.inputs[0].value
    err_rate = 1 - sum(iden_values)/ float(n_test)

    print 'test error rate by highest specific features activation sum %f %%' % (err_rate * 100.)

    f.write(str(err_rate * 100.))
    f.write('\n')

    iden = T.cast(test_set_y_values == labels3, 'int32')
    iden_values = iden.owner.inputs[0].value
    err_rate = 1 - sum(iden_values)/ float(n_test)

    print 'test error rate by highest thresholded activation count %f %%' % (err_rate * 100.)

    f.write(str(err_rate * 100.))
    f.write('\n')

    f.close()

    image = Image.fromarray(
        tile_raster_images(X=da.commonW.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('common_filters.png')

    # X=da.specW.get_value(borrow=True)
    for i in xrange(10):
	Simage = Image.fromarray(
        tile_raster_images(X=da.specW[i].eval().T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
        Simage.save('specific_filters_'+str(i)+'.png')


    os.chdir('../')


    #######################################
    # Histogram of the connection weights #
    #######################################

    '''w=da.W.eval()
    w_1d = w.reshape(da.n_visible*da.n_hidden)'''
    '''w_min=10
    w_max=0
    for i in xrange(da.n_input*da.n_hidden):
        if w_1d[i]<w_min:
            w_min=w_1d[i]
        if w_1d[i]>w_max:
            w_max=w_1d[i]
    print w_max
    print w_min'''

    
    '''w_prime=da.W_prime.eval()
    w_prime_1d = w_prime.reshape(da.n_visible*da.n_hidden)'''
    '''w_prime_min=10
    w_prime_max=0
    for i in xrange(da.n_output*da.n_hidden):
        if w_prime_1d[i]<w_prime_min:
            w_prime_min=w_prime_1d[i]
        if w_prime_1d[i]>w_prime_max:
            w_prime_max=w_prime_1d[i]
    print w_prime_max
    print w_prime_min'''


    '''num_pas=100
    w_pas=float(w_max-w_min)/num_pas
    bins_w=np.arange(w_min,w_max+w_pas,w_pas)
    
    w_prime_pas=float(w_prime_max-w_prime_min)/num_pas
    bins_w_prime=np.arange(w_prime_min,w_prime_max+w_prime_pas,w_prime_pas)'''
    
    '''plt.figure(1)
    plt.subplot(211)
    plt.hist(w_1d, bins=100)

    plt.subplot(212)
    plt.hist(w_prime_1d, bins=100)
    plt.show()'''

    
    ###############################
    # TRAINING Logistic Regression#
    ###############################
    # classify the values of the hidden layer
    '''com_hidden_output,spec_hidden_output=da.get_hidden_output(corruption_level=0., input=x)
    r_output = da.get_test_reconstructed_input(com_hidden_output,spec_hidden_output)
    wta_output= da.binary_lwta(r_output)
    #classifier = LogisticRegression(input=hidden_output, n_in=500, n_out=10)
    classifier = LogisticRegression(input=wta_output, n_in=784, n_out=10)
    cost_classifier = classifier.negative_log_likelihood(y)

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost_classifier, wrt=classifier.W)
    g_b = T.grad(cost=cost_classifier, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    classifier_learning_rate=0.13
    updates_classifier = [(classifier.W, classifier.W - classifier_learning_rate * g_W),
               (classifier.b, classifier.b - classifier_learning_rate * g_b)]

    train_classifier = theano.function(
        inputs=[index],
        outputs=cost_classifier,
        updates=updates_classifier,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_classifier = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_classifier = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
        
    print '... training the classifier'
    # early-stopping parameters
    n_epochs=1000
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_classifier(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_classifier(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_classifier(i)
                                   for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )'''


    


if __name__ == '__main__':
    for k in xrange(10):
        test_dA(k*5)
