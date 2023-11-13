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

import unittest

# Custom imports
import kitml_utils
import nlp_utils


class TestKITMolanStats(unittest.TestCase):


	def test_total_num_anns(self):
		'''Verify that the total number of annotations in KIT Motion Language
		dataset (released version) is same as described in the webpage here:
		https://motion-annotation.humanoids.kit.edu/dataset/
		NOTE: Paper describes an older version of the dataset.
		'''
		# KIT Motion-Language Dataset annotations
		anns = kitml_utils.load_kit_molan(l_re=['*annotations.json'])
		l_all_anns = list(flatten([anns[fid]['annotations'] for fid in anns]))

		# Stats from: https://motion-annotation.humanoids.kit.edu/dataset/
		self.assertEqual(3911, len(anns))
		self.assertEqual(6353, len(l_all_anns))


	def test_3files_for_all_seqs(self):
		'''Verify that there are 3 files of different types for all 3911 motions
		'''
		# KIT Motion-Language Dataset
		kitml = kitml_utils.load_kit_molan(l_re=['*'])
		c_type_files = [len(kitml[fid]) for fid in kitml]

		self.assertEqual(3911, len(kitml))
		self.assertEqual([3]*len(kitml), c_type_files)


	def test_all_motions_have_anns(self):
		'''Verify that all 3911 motions have at least one annotation.'''
		# KIT Motion-Language Dataset annotations
		anns = kitml_utils.load_kit_molan(l_re=['*annotations.json'])
		l_anns_per_fid = [len(anns[fid]['annotations']) for fid in anns]

		# Print some stats
		print('Mean +/- Std. dev. of # anns = {:.3f} +/- {:.3f}'.\
					format(np.mean(l_anns_per_fid), np.std(l_anns_per_fid)))
		l_nz_anns_per_fid = [c for c in l_anns_per_fid if c > 0]
		print('Mean +/- Std. dev. of # anns = {:.3f} +/- {:.3f}'.\
			format(np.mean(l_nz_anns_per_fid), np.std(l_nz_anns_per_fid)))

		# Test that there are 899 motions with 0 annotations
		self.assertEqual(899, Counter(l_anns_per_fid)[0])


if __name__ == '__main__':
	unittest.main()
