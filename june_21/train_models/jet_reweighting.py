# Adapted from https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py

import random
import numpy as np
import torch
from numpy.random import choice
#import torch.utils.data




class FlavEtaPtSampler(torch.utils.data.sampler.Sampler):
    
    def __init__(self, targets, bins_2d_eta_pt, indices = None, num_samples: int = None):
        
        self.indices = np.array(range(len(targets))) if indices is None else indices
        
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        
        self.eta_bins = bins_2d_eta_pt[0]-1
        self.pt_bins = bins_2d_eta_pt[1]-1
        
        self.labels = targets
                
        b_weights = np.load('/home/um106329/aisafety/may_21/absweights_b.npy')
        bb_weights = np.load('/home/um106329/aisafety/may_21/absweights_bb.npy')
        c_weights = np.load('/home/um106329/aisafety/may_21/absweights_c.npy')
        l_weights = np.load('/home/um106329/aisafety/may_21/absweights_l.npy')
        
        flavour_lookuptables = np.array([b_weights,bb_weights,c_weights,l_weights])
        #print(self.labels)
        #print(self.eta_bins)
        #print(self.pt_bins)
        #print(min(self.eta_bins))
        #print(max(self.eta_bins))
        #print(min(self.pt_bins))
        #print(max(self.pt_bins))
        #all_weights = []
        #for jet_id in self.indices:
        #    jet_weight = flavour_lookuptables[self.labels[jet_id]][self.eta_bins[jet_id]][self.pt_bins[jet_id]]
        #    all_weights.append(jet_weight)
        all_weights = flavour_lookuptables[self.labels,self.eta_bins,self.pt_bins]
        #self.weights = torch.DoubleTensor(all_weights)/sum(all_weights)
        self.weights = all_weights/sum(all_weights)
        #print(self.weights.unique())
        #print(len(self.weights.unique()))
        self.drawn = np.random.choice(self.indices, self.num_samples, replace=True, p=self.weights)
        
                
    def __iter__(self):
        # drawing samples in __iter__ might ultimately be the correct solution, but it takes too much time and is probably not that much more accurate
        # than keeping the selected indices for one script
        
        
        #return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))
        #return (self.indices[i] for i in random.choices(population=self.indices,weights=self.weights,k=self.num_samples))
        #return (self.indices[i] for i in np.random.choice(self.indices, self.num_samples, replace=True, p=self.weights))
        #return (choice(list_of_candidates, number_of_items_to_pick, p=probability_distribution))
        return (i for i in self.drawn)
    def __len__(self):
        return self.num_samples