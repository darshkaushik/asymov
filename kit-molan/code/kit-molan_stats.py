import sys, os, pdb
from os.path import join as ospj
from os.path import basename as ospb
import json
from glob import glob as glob
import re

from tqdm import tqdm

import numpy as np
from pandas.core.common import flatten
from collections import *

# Custom imports
import kitml_utils
import nlp_utils


class ComputeStats():

	def stats_num_words(self):
		'''Stats. about num. words in KIT Motion Langauge dataset.'''
		# KIT Motion-Language Dataset annotations
		anns = kitml_utils.load_kit_molan(l_re=['*annotations.json'])
		# NOTE: `flatten` gets rid of empty lists (missing annotations)
		l_all_anns = list(flatten([anns[fid]['annotations'] for fid in anns]))

		l_w, l_num_w, l_v, l_num_v, l_num_num_w = [], [], [], [], 0
		for ann in tqdm(l_all_anns):
			# Get (tokenized) words in each annotation
			words = nlp_utils.get_tok_words_in_string(ann)
			l_w += words
			l_num_w.append(len(words))
			verbs = nlp_utils.get_verbs_in_string(ann)
			l_v += verbs
			l_num_v.append(len(verbs))
			l_num_num_w += len(nlp_utils.get_number_words_in_string(ann))

		# Print stats.
		print('Total # words = ', len(l_w))
		print('Total # unique words = ', len(set(l_w)))
		print('Mean +/- Std. dev. # words per ann. = {:.3f} +/- {:.3f}'.\
									format(np.mean(l_num_w), np.std(l_num_w)))
		print('Mean +/- Std. dev. # verbs per ann. = {:.3f} +/- {:.3f}'.\
									format(np.mean(l_num_v), np.std(l_num_v)))
		print('Total # number words in dataset = ', l_num_num_w)


	def stats_verbs_per_motion(self):
		'''Stats. about num. verbs per motion in KIT Motion Langauge dataset.
		I.e., for each motion, obtain all unique actions. Then compute histogram
		of actions in the dataset.
		'''
		# KIT Motion-Language Dataset annotations
		anns = kitml_utils.load_kit_molan(l_re=['*annotations.json'])
		# verbs_of_int = ['wal', 'wale', 'left', 'puched', 'strumble', 'tae', 'puche']
		verbs_of_int = ['elbow', 'shoulder', 'knee']

		set_v_per_fid = {}
		for fid in anns:
			l_v_fid = []
			for ann in anns[fid]['annotations']:
				v_tup = nlp_utils.get_verbs_in_string(ann)
				verbs = [v[1] for v in v_tup]
				l_v_fid += verbs
				debug_v = set(verbs).intersection(set(verbs_of_int))
				if len(debug_v) > 0:
					print('Lemmatized "Verb": {0}, Sent: {1}'.format(
																debug_v, ann))
			set_v_per_fid[fid] = set(l_v_fid)

		# List of action verbs for each motion across dataset
		l_v = list(flatten([list(set_v_per_fid[fid]) for fid in set_v_per_fid]))
		print(Counter(l_v))


if __name__ == '__main__':
	ComputeStats()
