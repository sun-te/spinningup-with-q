import json
import joblib
import shutil
import numpy as np
import torch
import os.path as osp, time, atexit, os
import warnings
from spinup.utils.mpi_tools import proc_id, mpi_statistics_scalar

SAVE_FREQ = [10, 50, 100, 200, 300, 400, 500, 1000, 2000]

def pytorch_simple_save(prefix, output_dir, itr, saver_elements=None):
    """
    Saves the PyTorch model (or models).
    """
    if proc_id() == 0:
        fpath = 'pyt_save'
        fpath = osp.join(output_dir, fpath)
        fname = prefix + '_model' + ('%d' % itr if itr is not None else '') + '.pt'
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
            torch.save(saver_elements, fname)

def save_pi(output_dir, itr, paramenters=None):
    pytorch_simple_save("pi", output_dir, itr, paramenters)

def save_vf(output_dir, itr, paramenters=None):
    pytorch_simple_save("vf", output_dir, itr, paramenters)

def load_model(ac_model, fname, vf_itr=10, pi_itr=10):

    assert vf_itr in SAVE_FREQ and pi_itr in SAVE_FREQ, "Error: there is no such saving point"
    fname = fname+'/' if fname[-1] != '/' else fname
    pi_name = "pi_model"+str(pi_itr)+".pt"
    vf_name = "vf_model"+str(vf_itr)+".pt"
    pi_model_path = os.path.join(fname, pi_name)
    vf_model_path = os.path.join(fname, vf_name)
    import ipdb
    from copy import deepcopy

    old_params = [deepcopy(p) for p in ac_model.pi.parameters()]

    pi_model = torch.load(pi_model_path)
    # vf_model = torch.load(vf_model_path)
    pi_model.eval()
    # vf_model.eval()

    pi_pretrained_dict = pi_model.state_dict()
    pi_dict = ac_model.pi.state_dict()
    # load the pretrained model
    # 1. filter out unnecessary keys
    pi_pretrained_dict = {k: v for k, v in pi_pretrained_dict.items() if k in pi_dict}
    # 2. overwrite entries in the existing state dict
    pi_dict.update(pi_pretrained_dict)
    # 3. load the new state dict
    ac_model.pi.load_state_dict(pi_dict)

    # for i, p in enumerate(ac_model.pi.parameters()):
    #     print(torch.norm(p-old_params[i]))
    # print("______________________")
    # for p1, p2 in zip(pi_model.parameters(), ac_model.pi.parameters()):
    #     print(torch.norm(p1-p2))
    # ipdb.set_trace()



    return
