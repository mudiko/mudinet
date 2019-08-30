from tensor import Tensor
from typing import Iterator, NamedTuple
import numpy as np

Batch=NamedTuple("Batch", [("inputs",Tensor),("targets",Tensor)])

class DataIterator:
    def __call__(self,inputs:Tensor,targets:Tensor)->Iterator:
        raise NotImplementedError

class BatchIterator(DataIterator):
    def __init__(self, batchsize: int=32,shuffle: bool=True)->None:
        self.batchsize=batchsize
        self.shuffle=shuffle
    def __call__(self,inputs:Tensor,targets:Tensor)->Iterator:
        starts=np.arange(0,len(inputs),self.batchsize)
        if self.shuffle:
            np.random.shuffle(starts)
        for start in starts:
            end=start+self.batchsize
            batch_inputs=inputs[start:end]
            batch_targets=targets[start:end]
            yield Batch(batch_inputs,batch_targets)
