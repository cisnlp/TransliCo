#!/usr/bin/env python3
import codecs
import os.path
import os
from tqdm import tqdm


class Uroman:
	def __init__(self):
		pass

	def romanize(self, sentences, temp_path='./temp', lang=None):
		# sentences: a list of sentences
		uroman_path = "/mounts/Users/student/yihong/Documents/concept_align/uroman/bin/"

		# create parallel text
		in_path = temp_path + '/sentence' + ".txt"
		out_path = temp_path + '/sentence_roman' + ".txt"

		if not os.path.exists(temp_path):
			os.makedirs(temp_path)

		fa_file = codecs.open(in_path, "w", "utf-8")
		for sentence in sentences:
			fa_file.write(sentence + "\n")
		fa_file.close()

		if lang is None:
			os.system(uroman_path + "uroman.pl < {0} > {1} ".format(in_path, out_path))
		else:
			os.system(uroman_path + "uroman.pl -l {0} < {1} > {2} ".format(lang, in_path, out_path))

		romanize_sentences = []
		f1 = open(out_path, "r", encoding='utf-8')
		print("Transliteration starts ...")
		for line in tqdm(f1.readlines()):
			romanize_sentences.append(line.strip())

		os.system("rm {}".format(in_path))
		os.system("rm {}".format(out_path))

		return romanize_sentences
