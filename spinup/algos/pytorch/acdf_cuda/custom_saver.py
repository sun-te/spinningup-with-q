import json
import joblib
import shutil
import numpy as np
import torch
import os.path as osp, time, atexit, os
import warnings
from spinup.utils.mpi_tools import proc_id, mpi_statistics_scalar

def _pytorch_simple_save( itr=None):
    """
    Saves the PyTorch model (or models).
    """
    if proc_id() == 0:
        fpath = 'pyt_save'
        fpath = osp.join(self.output_dir, fpath)
        fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'
        fname = osp.join(fpath, fname)
        os.makedirs(fpath, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # We are using a non-recommended way of saving PyTorch models,
            # by pickling whole objects (which are dependent on the exact
            # directory structure at the time of saving) as opposed to
            # just saving network weights. This works sufficiently well
            # for the purposes of Spinning Up, but you may want to do
            # something different for your personal PyTorch project.
            # We use a catch_warnings() context to avoid the warnings about
            # not being able to save the source code.
            torch.save(self.pytorch_saver_elements, fname)