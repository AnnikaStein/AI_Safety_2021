# Source: https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
# Discussion / blog post: https://towardsdatascience.com/better-data-loading-20x-pytorch-speed-up-for-tabular-data-e264b9e34352

import gc
import torch

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False, n_data=None):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        #self.tensors[0] = self.tensors[0].float()
        # okay this is officially stupid, changing to float in order to save memory with the __next__ function
        
        #global train_inputs
        #global train_targets
        #global val_inputs
        #global val_targets
        #del train_inputs
        #del train_targets
        del tensors
        gc.collect()
        
        
        self.dataset_len = n_data
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        
    # select indices
    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            # original
            #batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
            # this one has to be changed in order to work with index_select --> therefore the call .float() inside the trainloop is not necessary anymore
            # nevermind I change it during inittialization
            #batch = tuple((torch.index_select(self.tensors[0].float(), 0, indices), torch.index_select(self.tensors[1], 0, indices)))
            # also not good, next try:
            batch = tuple(t[indices] for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch
        
    '''
    # shuffle in-place
    
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch
    '''
    
    
    def __len__(self):
        return self.n_batches