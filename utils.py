# Additional functions for the computational appendix of
# Efficient training of energy-based models via spin-glass control
# arXiv:1910.01592
#
# Authors: Alejandro Pozas-Kerstjens and Gorka Muñoz-Gil
#
# Requires: itertools for combinations
#           gc for garbage collection
#           pytorch as ML framework
#           tqdm for progress bar
# Last modified: Oct, 2019
import gc
import torch
from itertools import combinations
from torch import cat, max, min, ones, randn, Tensor, zeros
from torch.nn.functional import linear
from torch.utils.data import DataLoader
from tqdm import tqdm

class GradientRBM(torch.nn.Module):

    def __init__(self, n_visible=10, n_hidden=50, device=None, weights=None):
        '''Restricted Boltzmann machine with spin-like neurons to be trained via
        AutoML in PyTorch.

        Arguments:

            :param n_visible: The number nodes in the visible layer
            :type n_visible: int
            :param n_hidden: The number nodes in the hidden layer
            :type n_hidden: int
            :param device: Device where to perform computations. None is CPU.
            :type device: torch.device
            :param W: Optional parameter to specify the weights of the RBM
            :type W: torch.nn.Parameter
        '''
        super(GradientRBM, self).__init__()

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')

        if weights is not None:
            self.weights = torch.nn.Parameter(weights.to(self.device),
                                              requires_grad=True)
        else:
            self.weights = torch.nn.Parameter(Tensor(
                                               0.01 * randn(n_hidden, n_visible)
                                                     ).to(self.device),
                                              requires_grad=True)
    def free_energy(self, v):
        wx_b = linear(v, self.weights)
        a    = max(wx_b, -wx_b)

        hidden_term = (a + ((-wx_b - a).exp() + (wx_b - a).exp()).log()).sum(1)
        return -hidden_term

def batch_nll(machine, batch, all_confs, bs=65536):
    '''Compute the negative log-likelihood of a batch of training instances.
    This is employed when training the RBM with exact gradient descent.

    Arguments:

        :param machine: The model one wishes to train
        :type machine: torch.nn.Module
        :param batch: Batch of images for which the NLL will be computed
        :type batch: torch.Tensor
        :param all_confs: All possible configurations of the visible neurons
                          of the model
        :type all_confs: torch.Tensor
        :param bs: Internal batch size to compute the partition function of the
                   machine
        :type bs: int

        :returns float: Ground state energy of the model
    '''
    logZ = log_partition_function(machine, bs, all_confs)
    size = len(batch)
    fe   = machine.free_energy(batch).sum()
    return (fe + size * logZ).neg(), logZ

def create_bas(size):
    '''Generates the complete Bars and Stripes dataset of arbitrary size,
    in the spin notation (+1/-1 neurons)

    Arguments:

        :param size: Size of the images in the dataset
        :type size: int

        :returns dataset: The dataset
    '''
    all_configurations = []
    for i in range(size):
        if i != (size - 1):
            all_configurations += list(combinations(range(size), i+1))
    bars = [zeros((size, size)) - 1 for _ in range(len(all_configurations))]
    stripes = [zeros((size, size)) - 1 for _ in range(len(all_configurations))]
    for i, conf in enumerate(all_configurations):
        bars[i][:, conf] = 1
        stripes[i][conf, :] = 1

    dataset = bars + stripes
    return dataset

def create_bars_4x4():
    '''Create the 4x4-pixel Bars datasets, with flattened images

        :returns train_data: torch.Tensor of size (10,16) with training images
        :returns recon_train_data: torch.Tensor of size (10,16) with
                                   partially-erased training images
        :returns test_data: torch.Tensor of size (4,16) with testing images
        :returns recon_test_data: torch.Tensor of size (4,16) with
                                  partially-erased testing images
    '''
    train_1    = Tensor([[1, -1, -1, -1]] * 4).unsqueeze(0).view((-1, 16))
    train_2    = Tensor([[1, 1, -1, -1]] * 4).unsqueeze(0).view((-1, 16))
    train_3    = Tensor([[1, 1, 1, -1]] * 4).unsqueeze(0).view((-1, 16))
    train_4    = Tensor([[1, -1, 1, 1]] * 4).unsqueeze(0).view((-1, 16))
    train_5    = Tensor([[1, 1, -1, 1]] * 4).unsqueeze(0).view((-1, 16))
    train      = [train_1, train_2, train_3, train_4, train_5]
    inv_train  = [-image for image in train]
    train_data = cat(train + inv_train, 0)

    recon_train_data = train_data.clone()
    recon_train_data[:,4:] = 0

    test_1    = Tensor([[1, -1, 1, -1]] * 4).unsqueeze(0).view((-1, 16))
    test_2    = Tensor([[1, -1, -1, 1]] * 4).unsqueeze(0).view((-1, 16))
    test      = [test_1, test_2]
    inv_test  = [-image for image in test]
    test_data = cat(test + inv_test, 0)

    recon_test_data = test_data.clone()
    recon_test_data[:,4:] = 0

    return [train_data, recon_train_data, test_data, recon_test_data]

def gs_energy(machine, all_confs, batch_size=65536):
    '''Calculates the ground state energy of a model

    Arguments:

        :param machine: The model one wishes to sample from
        :type machine: torch.nn.Module
        :param all_confs: All possible configurations of the visible neurons
                          of the model
        :type all_confs: torch.Tensor
        :param bathc_size: Amount of images employued in each computation step

        :returns float: Ground state energy of the model
    '''
    energies = Tensor([]).to(machine.weights.device)
    for v in DataLoader(all_confs, batch_size=batch_size):
        fields = linear(v, machine.weights)
        energies = cat([energies, -fields.abs().sum(1)])
    return min(energies).item()

def logsumexp(tensor):
    '''Computes pointwise log(sum(exp())) for all elements in a torch tensor.
    The way of computing it without under- or overflows is through the
    log-sum-exp trick, namely computing
    log(1+exp(x)) = a + log(exp(-a) + exp(x-a))     with a = max(0, x)
    The function is adapted to be used in GPU if needed.

    Arguments:

        :param tensor: torch.Tensor
        :returns: torch.Tensor
    '''
    a = max(zeros(1).to(tensor.device), max(tensor))
    return a + (tensor - a).exp().sum().log()


def log_partition_function(rbm, batch_size, all_confs):
    '''Computes (via exact brute-force) the logarithm of the partition function
    of the Ising model defined by the weights of a Restricted Boltzmann Machine

    Arguments:

        :param rbm: Restricted Boltzmann Machine model
        :type rbm: :class:`ebm.models`
        :param batch_size: amount of samples used in every computation step
        :type batch_size: int
        :param all_confs: All possible configurations of the visible neurons
                          of the model
        :type all_confs: torch.Tensor

        :returns logZ: torch.Tensor with the logarithm of the partition function
    '''
    all_confs = DataLoader(all_confs.to(rbm.device), batch_size=batch_size)
    logsumexps = Tensor([]).to(rbm.device)
  #  for batch in tqdm(all_confs, desc='Computing partition function'):
    for batch in all_confs:
        logsumexps = cat([logsumexps, logsumexp(rbm.free_energy(batch).neg())])
    logZ = logsumexp(logsumexps)
    gc.collect()
    return logZ
