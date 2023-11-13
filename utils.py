#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

import os
from pathlib import Path
import numpy as np
import pdb
import json
import pickle


"""
Script containing many common utils
"""

def read_pickle(filepath):
    '''Read the pickle file and return its contents'''
    data = None
    with open(filepath, 'rb') as infile:
        data = pickle.load(infile)
    return data


def read_json(filepath):
    '''Read the json file and return its contents'''
    # TODO: Test
    data = None
    with open(filepath, 'rb') as infile:
        data = json.load(infile)
    return data


def _dir_exists(filepath):
    if not Path(filepath).parent.is_dir():
        raise FileNotFoundError(f'Dirpath for {filepath} does not exist!')
    return True


def _file_exists(filepath, overwrite):
    if os.path.exists(filepath):
        if not overwrite:
            print(f'Warning: Overwriting existing file: {filepath}')
        else:
            raise FileExistsError(f'File already exists! {filepath}')
    return True


def write_json(data, filepath, readable=False, overwrite=False):
    '''Write the contents into json file
    '''
    # TODO: Test

    # Make sure the file directory exists
    _dir_exists(filepath)

    # Make sure that the file doesn't already exist
    _file_exists(filepath, overwrite)

    # Save data to disk at 'filepath'
    with open(filepath, 'w') as outfile:
        if readable:
            json.dump(outfile, indent=4, separators=(',', ':'))
        else:
            json.dump(outfile)
    return 1


def write_pickle(data, filepath, overwrite=False):
    '''Write the contents into pickle file
    '''
    # TODO: Test

    # Make sure the file directory exists
    _dir_exists(filepath)

    # Make sure that the file doesn't already exist
    _file_exists(filepath, overwrite)

    # Save data to disk at 'filepath'
    with open(filepath, 'wb') as outfile:
        pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    return 1


def write_textf(data, filepath, overwrite=False):
    '''Write the string contents into text file
    '''
    # TODO: Test

    # Make sure the file directory exists
    _dir_exists(filepath)

    # Make sure that the file doesn't already exist
    _file_exists(filepath, overwrite)

    # Save text file to disk at 'filepath'
    with open(filepath, 'w') as outfile:
        outfile.write(data)
    return 1


def read_textf(filepath, ret_type='str'):
    '''Read the text file and return its contents.

    Args:
        ret_type <str>: {'str', 'list'}. 'str' returns read of file contents.
    'list' returns list of strings split by newline (readlines).

    '''
    data = None
    with open(filepath, 'r') as f:
        if 'str' == ret_type:
            data = f.read()
        elif 'list' == ret_type:
            data = f.read().splitlines()
    return data


def int2padstr(data, strlen):
    '''Given a positive int (or float or string of an int), return a
0-padded string of desired length.
    Args:
        data {<int>, <str>, <float>} : The data to be padded.
        strlen <int> : Total length of string to be returned.
    Return:
        padstr <str>: 0-padded string of desired length containing data.
    '''

    # Type checks
    if type(data) is float:
        if not data.is_integer():
            raise TypeError(f'Expected whole number but got a fraction: {data}')
        else:
            data = str(int(data))  # Cast float --> str (w/o the decimal part)

    elif type(data) is str:
        data = int(data)

    # Verify if pad-length is greater than data-length
    if len(str(data)) > strlen:
        raise ValueError('Pad-length is too short. Total length of padded string should be >= data length.')

    # Pad the <int> data with 0s
    padstr = str(data).zfill(strlen)
    return padstr


def read_cfg_w_hydra(cfg_rootdir, cfg_name):
    import hydra
    hydra.initialize(version_base=None, config_path=cfg_rootdir)
    cfg = hydra.compose(config_name=cfg_name)
    return cfg


def get_ext(fname):
    '''Return extension of filename if exists. Else return None'''
    ext = None
    l_fn = Path(fname).name.split('.')
    if len(l_fn) > 1:
        ext = l_fn[-1]
    return ext


def read_cfg_w_omega(cfg_rootdir, cfg_name):
    import omegaconf
    fn = cfg_name+'.yaml' if get_ext(cfg_name) is None else cfg_name
    cfg = omegaconf.OmegaConf.load(Path(cfg_rootdir, f'{fn}'))
    return cfg


