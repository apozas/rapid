# Code for
# Efficient training of energy-based models via frustration reduction
# arXiv:
#
# Authors: Alejandro Pozas-Kerstjens and Gorka Muñoz-Gil
#
# Requires: ebm-torch for ML models   (https://www.github.com/apozas/ebm-torch)
#           math for mathematical operations
#           pytorch as ML framework
#           tqdm for progress bar
# Last modified: Oct, 2019
import math
from ebm.models import RBM
from ebm.optimizers import SGD, Adam, outer_product
from ebm.samplers import ContrastiveDivergence as cd
from torch import cat, einsum, max, mm, randint, rand_like, sigmoid, sign,     \
                  sqrt, tanh, zeros_like
from torch.nn.functional import linear, dropout
from torch.nn import Parameter
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class RBM_pm(RBM):

    def __init__(self, n_visible=10, n_hidden=50, sampler=None, optimizer=None,
                 device=None, weights=None):
        '''Restricted Boltzmann machine with spin-like neurons (their allowed
           values are +1/-1) instead of binary (0/1) neurons, and with no
           biases.
           
        Arguments:
           
            :param n_visible: The number nodes in the visible layer
            :type n_visible: int
            :param n_hidden: The number nodes in the hidden layer
            :type n_hidden: int
            :param sampler: Method used to draw samples from the model
            :type sampler: :class:`ebm.samplers`
            :param optimizer: Optimizer used for parameter updates
            :type optimizer: :class:`ebm.optimizers`
            :param device: Device where to perform computations. None is CPU.
            :type device: torch.device
            :param W: Optional parameter to specify the weights of the RBM
            :type W: torch.nn.Parameter
            :param vbias: Optional parameter to specify the visible biases of
                          the RBM
            :type vbias: torch.nn.Parameter
            :param hbias: Optional parameter to specify the hidden biases of
                          the RBM
            :type hbias: torch.nn.Parameter
        '''
        super().__init__(n_visible, n_hidden, sampler, optimizer, device, weights)
    
    def free_energy(self, v):
        '''Computes the free energy for a given state of the visible layer.

        Arguments:

            :param v: The state of the visible layer of the RBM
            :type v: torch.Tensor

            :returns: torch.Tensor
        '''
        
        wx_b = linear(v, self.weights, self.hbias)
        
        # Fancy (and overflow-resistant) way of computing log(2cosh(x))
        a = max(wx_b, -wx_b)
        hidden_term = (a + ((-wx_b - a).exp() + (wx_b - a).exp()).log()).sum(1)
        return -hidden_term

    def train(self, input_data):
        '''Trains the RBM.

        Arguments:

            :param input_data: Batch of training points
            :type input_data: torch.utils.data.DataLoader
        '''
        for batch in tqdm(input_data, desc=('Epoch ' +
                                            str(self.optimizer.epoch + 1))):
            
            sample_data = batch.float()
            
            vpos = sample_data
            vneg = self.sampler.get_negative_sample(vpos, self.weights,
                                                   self.vbias, self.hbias)
            W_update, _, _ = \
                             self.optimizer.get_updates(vpos, vneg,
                                                self.weights, self.vbias, self.hbias)
            self.weights += W_update
             
        self.optimizer.epoch += 1


class RA_RBM(RBM_pm):

    def __init__(self, n_visible=100, n_hidden=50, K=50,
                 optimizer=None, device=None, xi=None):
        '''RBM where the weights are computed through the method of Restricted
           Axon. The weights are computed from low-energy patterns, and the
           parameters to be optimized is the patterns themselves
        
        Arguments:
            
            :param n_visible: The number nodes in the visible layer
            :type n_visible: int
            :param n_hidden: The number nodes in the hidden layer
            :type n_hidden: int
            :param K: The number of patterns from which the weights are computed
            :type K: int
            :param sampler: Method used to draw samples from the model
            :type sampler: :class:`samplers`
            :param optimizer: Optimizer used for parameter updates
            :type optimizer: :class:`optimizers`
            :param device: Device where to perform computations. None is CPU.
            :type device: torch.device
            :param xi: Optional parameter to specify the initial patterns
            :type xi: torch.nn.Parameter
            :param vbias: Optional parameter to specify the visible biases of
                          the RBM
            :type vbias: torch.nn.Parameter
            :param hbias: Optional parameter to specify the hidden biases of
                          the RBM
            :type hbias: torch.nn.Parameter'''
        super().__init__(n_visible, n_hidden, 'None', optimizer, device)
        self.K         = K
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        
        if xi is not None:
            self.xi = xi
        else:
            self.xi = Parameter((2 * randint(0,
                                             2,
                                        (self.K, self.n_hidden + self.n_visible)
                                             ) - 1).float().to(self.device))
        for param in self.parameters():
            param.requires_grad = False  

        self.get_params()
            
    def get_params(self):
        '''Computes the weight matrix of the RBM from the patterns'''
        vis  = self.xi[:, :self.n_visible]
        hidd = self.xi[:, self.n_visible:]
        self.weights.data = (outer_product(hidd, vis).sum(0)).to(self.device)
        self.weights.data /= math.sqrt(self.K)
        
    def train(self, input_data):
        '''Trains the RBM.

        Arguments:

            :param input_data: Batch of training points
            :type input_data: torch.utils.data.DataLoader
        '''
        
        for batch in tqdm(input_data, desc=('Epoch ' +
                                            str(self.optimizer.epoch + 1))):

            sample_data = batch.float()

            vpos = sample_data
            # Get negative phase from the patterns
            vneg = sign(self.xi[:, :vpos.shape[1]])

            xi_update = self.optimizer.get_updates(vpos, vneg, self.xi)
            self.xi += xi_update

            self.get_params()
        
        # Renormalize after the training epoch has concluded
        self.xi.data = sign(self.xi)
        self.get_params()
        self.optimizer.epoch += 1    

# -----------------------------------------------------------------------------
# Samplers
# -----------------------------------------------------------------------------
class ContrastiveDivergence_pm(cd):

    def __init__(self, k, dropout=0):
        '''Obtains samples of RBM models via Gibbs iteration of predetermined
        initial visible configurations, using spin notation (+1,-1).
        
        Arguments:
        
            :param k: The number of iterations in CD-k
            :type k: int
            :param dropout: Optional parameter, fraction of neurons in the
                            previous layer that are not taken into account when
                            getting a sample.
            :type dropout: float
        '''
        super().__init__(k, dropout, continuous_output=False)

    def get_h_from_v(self, v, W, hbias):
        h_probs = self._propup(v, W, hbias)
        h_sample = 2 * h_probs.bernoulli() - 1
        return h_sample

    def get_v_from_h(self, h, W, vbias):
        v_probs = self._propdown(h, W, vbias)
        v_sample = 2 * v_probs.bernoulli() - 1
        return v_sample

    def _propdown(self, h, W, vbias):
        pre_sigmoid_activation = linear(dropout(h, self.dropout), W.t(), vbias)
        return sigmoid(2 * pre_sigmoid_activation) 

    def _propup(self, v, W, hbias):
        pre_sigmoid_activation = linear(dropout(v, self.dropout), W, hbias)
        return sigmoid(2 * pre_sigmoid_activation)


class PersistentContrastiveDivergence_pm(ContrastiveDivergence_pm):

    def __init__(self, k, n_chains=0, dropout=0):
        '''Obtains samples of RBM models via Gibbs iteration of fantasy
        particles, using spin notation (+1,-1).
        
        Arguments:
        
            :param k: The number of iterations in PCD-k
            :type k: int
            :param n_chains: The number of fantasy particles for negative phase
            :type k: int
            :param dropout: Optional parameter, fraction of neurons in the
                            previous layer that are not taken into account when
                            getting a sample.
            :type dropout: float
        '''
        super().__init__(k, dropout)
        self.n_chains = n_chains
        self.first_call = True

    def get_negative_sample(self, v0, W, vbias, hbias):
        if self.first_call:
            if self.n_chains <= 0:
                self.markov_chains = rand_like(v0)
            else:
                self.markov_chains = 2 * randint(0,
                                                 2,
                                                 (self.n_chains,) + v0.size()[1:]    # size
                                                 ).float().to(v0.device) - 1
            self.first_call = False
        for _ in range(self.k):
            h = self.get_h_from_v(self.markov_chains, W, hbias)
            v = self.get_v_from_h(h, W, vbias)
            self.markov_chains = v
        return v

# -----------------------------------------------------------------------------
# Optimizers
# -----------------------------------------------------------------------------
class Adam_pm(Adam):

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        '''Update the value of the RBM weights via the Adam algorithm

        Arguments:

            :param learning_rate: Learning rate
            :type learning_rate: float
            :param beta1: Adam parameter, regularization of parameters
            :type beta1: float
            :param beta2: Adam parameter, regularization of parameter norms
            :type beta2: float
            :param eps: Adam parameter, regularization of divergences
            :type eps: float
        '''
        super().__init__(learning_rate, beta1, beta2, eps)

    def get_updates(self, vpos, vneg, W, vbias, hbias):
        if self.first_call:
            self.m_W = zeros_like(W)
            self.m_v = zeros_like(vbias)
            self.m_h = zeros_like(hbias)
            self.v_W = zeros_like(W)
            self.v_v = zeros_like(vbias)
            self.v_h = zeros_like(hbias)
            self.first_call = False

        hpos = tanh(linear(vpos, W, hbias))
        hneg = tanh(linear(vneg, W, hbias))
        deltaW = (outer_product(hpos, vpos).mean(0)
                  - outer_product(hneg, vneg).mean(0))
        deltah = hpos.mean(0) - hneg.mean(0)
        deltav = vpos.mean(0) - vneg.mean(0)

        self.m_W *= self.beta1
        self.m_W += (1 - self.beta1) * deltaW
        self.m_v *= self.beta1
        self.m_v += (1 - self.beta1) * deltav
        self.m_h *= self.beta1
        self.m_h += (1 - self.beta1) * deltah

        self.v_W *= self.beta2
        self.v_W += (1 - self.beta2) * deltaW * deltaW
        self.v_v *= self.beta2
        self.v_v += (1 - self.beta2) * deltav * deltav
        self.v_h *= self.beta2
        self.v_h += (1 - self.beta2) * deltah * deltah

        mnorm_W = self.m_W / (1 - self.beta1 ** (self.epoch + 1))
        mnorm_v = self.m_v / (1 - self.beta1 ** (self.epoch + 1))
        mnorm_h = self.m_h / (1 - self.beta1 ** (self.epoch + 1))

        vnorm_W = self.v_W / (1 - self.beta2 ** (self.epoch + 1))
        vnorm_v = self.v_v / (1 - self.beta2 ** (self.epoch + 1))
        vnorm_h = self.v_h / (1 - self.beta2 ** (self.epoch + 1))

        self.weights_update     = self.learning_rate * mnorm_W / (sqrt(vnorm_W) + self.eps)
        self.vbias_update = self.learning_rate * mnorm_v / (sqrt(vnorm_v) + self.eps)
        self.hbias_update = self.learning_rate * mnorm_h / (sqrt(vnorm_h) + self.eps)

        return self.weights_update, self.vbias_update, self.hbias_update


class SGD_xi(SGD):
    
    def __init__(self, learning_rate, momentum=0, weight_decay=0):
        '''Update the value of the pattern units via Stochastic Gradient Descent
        
        Arguments:

            :param learning_rate: Learning rate
            :type learning_rate: float
            :param weight_decay: Weight decay parameter, to prevent overfitting
            :type weight_decay: float
            :param momentum: Momentum parameter, for improved learning
            :type momentum: float
        '''
        super().__init__(learning_rate, momentum, weight_decay)

    def get_params(self, xi, n_visible):
        '''Computes the neuron connections (axons) of the RBM from the patterns
        using the Hebbian rule
        
        Arguments:

            :param xi: Patterns from which to compute the weights
            :type xi: torch.Tensor
            :param n_visible: Number of visible neurons in the model
            :type n_visible: int
        '''
        vis = xi[:, :n_visible]
        hidd = xi[:, n_visible:]
        W = (mm(hidd.t(), vis) / math.sqrt(xi.shape[0])).to(xi.device)
        return W

    def get_updates(self, vpos, vneg, xi):
        '''Obtain the parameter updates
        
        Arguments:

            :param vpos: Batch of samples from the training set
            :type vpos: torch.Tensor
            :param vneg: Batch of samples drawn from the model
            :type vneg: torch.Tensor
            :param xi: Patterns from which to compute the weights
            :type xi: torch.Tensor
        '''
        if self.first_call:
            self.n_vis      = vpos.shape[1]
            self.pos_batch  = vpos.shape[0]
            self.neg_batch  = vneg.shape[0]
            self.K          = xi.shape[0]
            self.xi_update  = zeros_like(xi)
            self.first_call = False

        self.xi_update *= self.momentum
        self.xi_update -= self.learning_rate * self.weight_decay * xi
        
        W = self.get_params(xi, self.n_vis)
        xi_vis = xi[:, :self.n_vis]
        xi_hid = xi[:, self.n_vis:]
        deltaxi_v = (einsum('bj,bk->kj', (vpos, mm(tanh(mm(vpos, W.t())), xi_hid.t()))) / self.pos_batch
                   - einsum('bj,bk->kj', (vneg, mm(tanh(mm(vneg, W.t())), xi_hid.t()))) / self.neg_batch) / math.sqrt(self.K)
        deltaxi_h = (einsum('bj,kj,ba->ka',(vpos, xi_vis, tanh(mm(vpos, W.t())))) / self.pos_batch
                   - einsum('bj,kj,ba->ka',(vneg, xi_vis, tanh(mm(vneg, W.t())))) / self.neg_batch) / math.sqrt(self.K)
        deltaxi = cat([deltaxi_v, deltaxi_h], 1)
        self.xi_update.data += self.learning_rate * deltaxi

        return self.xi_update
