#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.


import pdb
import spacy
nlp = spacy.load('en_core_web_sm')

"""
Scripts that are useful while processing text
"""


def get_verbs_in_string(str):
	'''Given a string (multiple sents), return a list of verbs in the string.'''
	l_v = []
	tags = nlp(str)
	for tok in tags:
		if 'VERB' == tok.pos_:
			l_v.append((tok.text, tok.lemma_))
	return l_v


def get_tok_words_in_string(str):
	'''Given a string (multiple sents), return the # words in the string.'''
	l_w = []
	tags = nlp(str)
	for tok in tags:
		if tok.pos_ not in ['SYM', 'PUNCT']:  # Ignore non-word tokens
			l_w.append(tok.text)
	return l_w


def get_number_words_in_string(str):
	'''Given a string (multiple sents), return the number words in the string.
	E.g. for number words = {"two", "2", ...}.
	'''
	l_nw = []
	tags = nlp(str)
	for tok in tags:
		if 'NUM' == tok.pos_:
			l_nw.append(tok.text)
	return l_nw
