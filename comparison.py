# Example of usage: RAPID vs. other training methods. Part of
# Efficient training of energy-based models via spin-glass control
# arXiv:1910.01592
#
# Authors: Alejandro Pozas-Kerstjens and Gorka Muñoz-Gil
#
# Requires: ebm-torch for ML models   (https://www.github.com/apozas/ebm-torch)
#           itertools for Cartesian product
#           gc for garbage collection
#           numpy for numerics
#           pytorch as ML framework
#           matplotlib for plots
#           tqdm for progress bar
# Last modified: Jul, 2020
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from itertools import product
from rapid import ContrastiveDivergence_pm as CD, \
                  PersistentContrastiveDivergence_pm as PCD
from rapid import Adam_pm, SGD_xi
from rapid import RBM_pm, RA_RBM
from torch.optim import Adam
from utils import batch_nll, GradientRBM, create_bars_4x4, gs_energy

#------------------------------------------------------------------------------
# Auxiliary functions
#------------------------------------------------------------------------------
def gs_accessibility(machine, gs, steps=5, n_chains=10):
    '''Calculates how hard is for Gibbs iterations to get low energy states, by
    comparing the average energy of the last of 'steps' Gibbs iterations to the
    ground state energy

    Arguments:

        :param machine: The model one wishes to sample from
        :type machine: torch.nn.Module
        :param gs: The energy of the ground state of 'machine'
        :type gs: float
        :param steps: Number of Gibbs steps before retrieval
        :type steps: int
        :param n_chains: Number of samples run in parallel
        :type n_chains: int

        :returns float: energy relative to ground state energy
    '''
    vis = torch.randint(0, 2, (n_chains, machine.weights.shape[1]))
    vis = (2 * vis  - 1).float().to(machine.device)
    for _ in range(steps):
        hid = sampler.get_h_from_v(vis, machine.weights, hbias)
        vis = sampler.get_v_from_h(hid, machine.weights, vbias)
    energy = torch.min(-torch.einsum('bi,ji,bj->b',
                                     (vis, machine.weights, hid))).item()
    return energy / gs

#------------------------------------------------------------------------------
# Parameter choices
#------------------------------------------------------------------------------
hidd     = 1000        # Number of nodes in the hidden layer
rapid_lr = 0.02        # Learning rate for RAPID
other_lr = 0.001       # Learning rate for other methods
epochs   = 300         # Training epochs
K        = 8           # Number of patterns for RA model
k        = 10          # Gibbs steps for CD-PCD
bs_rapid = 3           # Batch size for RAPID
bs_other = 1           # Batch size for other methods
gpu      = True        # Use of GPU

#------------------------------------------------------------------------------
# Data preparation
#------------------------------------------------------------------------------
device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

all_datasets = create_bars_4x4()
for _ in all_datasets:
    dataset = all_datasets.pop(0)
    dataset = dataset.to(device)
    all_datasets.append(dataset)

train_set, recon_train, test_set, recon_test = all_datasets

vis = len(train_set[0])

all_confs = torch.Tensor(list(product([-1, 1], repeat=vis))).to(device)

# Extra parameters needed for the samplers
hbias = torch.zeros(hidd).to(device)
vbias = torch.zeros(vis).to(device)

# -----------------------------------------------------------------------------
# Construct Models
# -----------------------------------------------------------------------------
sampler = CD(k=k)      # Generic sampler
cd      = CD(k=k)
pcd     = PCD(k=k, n_chains=2048)
opt_ra  = SGD_xi(rapid_lr)
opt_cd  = Adam_pm(other_lr)
opt_pc  = Adam_pm(other_lr)

rbm_ra = RA_RBM(n_visible=vis,
                n_hidden=hidd,
                K=K,
                optimizer=opt_ra,
                device=device
                ).to(device)
rbm_cd = RBM_pm(n_visible=vis,
                n_hidden=hidd,
                sampler=cd,
                optimizer=opt_cd,
                device=device
                ).to(device)
rbm_pc = RBM_pm(n_visible=vis,
                n_hidden=hidd,
                sampler=pcd,
                optimizer=opt_pc,
                device=device
                ).to(device)
rbm_ex = GradientRBM(n_visible=vis,
                     n_hidden=hidd,
                     device=device
                     ).to(device)

opt_ex = Adam(rbm_ex.parameters(), lr=other_lr)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
access_pc = []
access_cd = []
access_ex = []
access_ra = []
hd_train_cd = []
hd_train_pc = []
hd_train_ex = []
hd_train_ra = []
hd_test_cd = []
hd_test_pc = []
hd_test_ex = []
hd_test_ra = []
min_pc_energy = []
min_cd_energy = []
min_ra_energy = []

machines = [rbm_ra, rbm_ex, rbm_cd, rbm_pc]

for epoch in range(epochs):
    train_loader_rapid = torch.utils.data.DataLoader(train_set,
                                                     batch_size=bs_rapid,
                                                     shuffle=True)
    train_loader_other = torch.utils.data.DataLoader(train_set,
                                                     batch_size=bs_other,
                                                     shuffle=True)
    # Training
    rbm_cd.train(train_loader_other)
    rbm_pc.train(train_loader_other)
    rbm_ra.train(train_loader_rapid)
    if rbm_ra.optimizer.learning_rate > 1e-5:
        rbm_ra.optimizer.learning_rate *= 0.96
    for batch in tqdm(train_loader_other, desc=('Epoch ' + str(epoch + 1))):
        cost, _ = batch_nll(rbm_ex, batch, all_confs)
        opt_ex.zero_grad()
        cost.backward()
        opt_ex.step()

    # Ground state energy
    ground_pc = gs_energy(rbm_pc, all_confs)
    ground_cd = gs_energy(rbm_cd, all_confs)
    ground_ex = gs_energy(rbm_ex, all_confs)
    ground_ra = gs_energy(rbm_ra, all_confs)

    # Accessibility of GS
    pc_gs_acc = gs_accessibility(rbm_pc, ground_pc, steps=k)
    cd_gs_acc = gs_accessibility(rbm_cd, ground_cd, steps=k)
    ex_gs_acc = gs_accessibility(rbm_ex, ground_ex, steps=k)
    ra_gs_acc = gs_accessibility(rbm_ra, ground_ra, steps=k)
    access_pc.append(pc_gs_acc)
    access_cd.append(cd_gs_acc)
    access_ex.append(ex_gs_acc)
    access_ra.append(ra_gs_acc)

    # Quality of sampling
    # PCD: Take energy of the lowest-energy chain of the PCD training
    chains_hidd = sampler.get_h_from_v(rbm_pc.sampler.markov_chains,
                                       rbm_pc.weights, hbias)
    min_chain_energy = torch.min(-torch.einsum('bi,ji,bj->b',
                     (rbm_pc.sampler.markov_chains, rbm_pc.weights, chains_hidd)
                                               )).item()
    min_pc_energy.append(min_chain_energy / ground_pc)

    # CD: Take all training set, do k Gibbs steps, and take the lowest-energy.
    # We repeat 10 times to avoid fluctuations
    smallest_cd_energies = []
    for idx in range(10):
        for gibbs_step in range(k):
            if gibbs_step == 0:
                sample_vis = train_set.clone()
            sample_hidd = sampler.get_h_from_v(sample_vis,
                                               rbm_cd.weights,
                                               hbias)
            sample_vis = sampler.get_v_from_h(sample_hidd,
                                              rbm_cd.weights,
                                              vbias)
        sample_hidd = sampler.get_h_from_v(sample_vis, rbm_cd.weights, hbias)
        smallest_cd_energies.append(torch.min(-torch.einsum('bi,ji,bj->b',
                                      (sample_vis, rbm_cd.weights, sample_hidd))
                                              ).item())
    min_cd_energy.append(np.min(smallest_cd_energies) / ground_cd)

    # RAPID: Take the lowest-energy pattern
    patt_vis, patt_hidd = rbm_ra.xi[:, :vis], rbm_ra.xi[:, vis:]
    smallest_pid_energy = torch.min(-torch.einsum('bi,ji,bj->b',
                                          (patt_vis, rbm_ra.weights, patt_hidd))
                                    ).item()
    min_ra_energy.append(smallest_pid_energy / ground_ra)

    # Hamming distances. Compute averages over 100 reconstruction instances
    hamming_train = [[], [], [], []]
    for complete, partial in zip(train_set, recon_train):
        to_reconstruct = partial.unsqueeze(0).repeat(100, 1).to(device)
        for machine_idx, machine in enumerate(machines):
            gibbs_vis  = to_reconstruct.clone()
            gibbs_hidd = sampler.get_h_from_v(gibbs_vis,
                                              machine.weights,
                                              hbias)
            gibbs_vis  = sampler.get_v_from_h(gibbs_hidd,
                                              machine.weights,
                                              vbias)
            hamming_train[machine_idx].append(
                    (
                     gibbs_vis - complete.unsqueeze(0).repeat(100, 1).to(device)
                     ).abs().mean().item() / 2)

    hd_train_ra.append(np.array(hamming_train[0]).mean())
    hd_train_ex.append(np.array(hamming_train[1]).mean())
    hd_train_cd.append(np.array(hamming_train[2]).mean())
    hd_train_pc.append(np.array(hamming_train[3]).mean())

    hamming_test = [[], [], [], []]
    for complete, partial in zip(test_set, recon_test):
        to_reconstruct = partial.unsqueeze(0).repeat(100, 1).to(device)
        for machine_idx, machine in enumerate(machines):
            gibbs_vis  = to_reconstruct.clone()
            gibbs_hidd = sampler.get_h_from_v(gibbs_vis,
                                              machine.weights,
                                              hbias)
            gibbs_vis  = sampler.get_v_from_h(gibbs_hidd,
                                              machine.weights,
                                              vbias)
            hamming_test[machine_idx].append(
                    (
                     gibbs_vis - complete.unsqueeze(0).repeat(100, 1).to(device)
                     ).abs().mean().item() / 2)

    hd_test_ra.append(np.array(hamming_test[0]).mean())
    hd_test_ex.append(np.array(hamming_test[1]).mean())
    hd_test_cd.append(np.array(hamming_test[2]).mean())
    hd_test_pc.append(np.array(hamming_test[3]).mean())
    print('train', hd_train_ra[-1], 'test', hd_test_ra[-1])

#-----------------------------------------------------------------------------
#Save all information
#-----------------------------------------------------------------------------
gc.collect()
np.savetxt('gibbs_access_pc.txt', access_pc)
np.savetxt('gibbs_access_cd.txt', access_cd)
np.savetxt('gibbs_access_ex.txt', access_ex)
np.savetxt('gibbs_access_ra.txt', access_ra)
np.savetxt('pattern_access_pc.txt', min_pc_energy)
np.savetxt('pattern_access_cd.txt', min_cd_energy)
np.savetxt('pattern_access_ra.txt', min_ra_energy)
np.savetxt('hd_train_pc.txt', hd_train_pc)
np.savetxt('hd_train_cd.txt', hd_train_cd)
np.savetxt('hd_train_ex.txt', hd_train_ex)
np.savetxt('hd_train_ra.txt', hd_train_ra)
np.savetxt('hd_test_pc.txt', hd_test_pc)
np.savetxt('hd_test_cd.txt', hd_test_cd)
np.savetxt('hd_test_ex.txt', hd_test_ex)
np.savetxt('hd_test_ra.txt', hd_test_ra)

# -----------------------------------------------------------------------------
# Plot Figures 1 and 2
# -----------------------------------------------------------------------------

x = list(range(1, epochs+1))

# Figure 1
plt.semilogx(x, min_ra_energy, label='RAPID', color='tab:blue')
plt.semilogx(x, min_pc_energy, label='PCD-10', color='tab:green')
plt.semilogx(x, min_cd_energy, label='CD-10', color='tab:orange')
plt.legend(loc=4, fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Method GS accessibility')
ax = plt.gca()
ax.get_legend().get_title().set_fontsize('14')
plt.savefig('pattern_access.pdf', bbox_inches='tight')
plt.clf()

plt.semilogx(x, access_ra, label='RAPID', color='tab:blue')
plt.semilogx(x, access_pc, label='PCD-10', color='tab:green')
plt.semilogx(x, access_cd, label='CD-10', color='tab:orange')
plt.semilogx(x, access_ex, label='Exact', color='tab:red')
plt.legend(title='Training method', loc=(0.02,0.29), fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Gibbs GS accessibility')
ax = plt.gca()
ax.get_legend().get_title().set_fontsize('14')
plt.savefig('gibbs_access.pdf', bbox_inches='tight')
plt.clf()

# Figure 2
plt.semilogx(x, hd_train_ra, label='RAPID', color='tab:blue')
plt.semilogx(x, hd_train_pc, label='PCD-10', color='tab:green')
plt.semilogx(x, hd_train_cd, label='CD-10', color='tab:orange')
plt.semilogx(x, hd_train_ex, label='Exact', color='tab:red')
plt.legend(title='Training method', loc=1, fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Hamming distance')
plt.title('Training set')
ax = plt.gca()
ax.get_legend().get_title().set_fontsize('14')
plt.savefig('hd_train.pdf', bbox_inches='tight')
plt.clf()

plt.semilogx(x, hd_test_ra, label='RAPID', color='tab:blue')
plt.semilogx(x, hd_test_pc, label='PCD-10', color='tab:green')
plt.semilogx(x, hd_test_cd, label='CD-10', color='tab:orange')
plt.semilogx(x, hd_test_ex, label='Exact', color='tab:red')
plt.legend(title='Training method', loc=(0.31,0.21), fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Hamming distance')
plt.title('Test set')
ax = plt.gca()
ax.get_legend().get_title().set_fontsize('14')
plt.savefig('hd_test.pdf', bbox_inches='tight')
plt.clf()
