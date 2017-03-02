# -*- encoding:utf-8 -*-

import os
import re
import glob
import json
import time
import fileinput
from collections import defaultdict

from config import *
from constant import *


def transform_to_tsv(query={}, filename='../data/lung_cancer.tsv'):
	'''
		transform to tsv (text preprocessing)
	'''
	import nltk
	wnl = nltk.WordNetLemmatizer()
	normalize = lambda raw:u' '.join([wnl.lemmatize(t).lower() for t in nltk.word_tokenize(raw)])
	with open(filename,'w') as outfile:
		for entry in db['articles'].find(query):
			outfile.write((u'\t'*3).join([normalize(entry.get(field,'').replace('\t','')) for field in ('dc:title','dc:description')]).encode('utf-8')+'\n')


def get_worddict(filename='../data/lung_cancer.tsv', basedir=r'../data/thesaurus/*.txt', savefile='worddict.json', threshold=lambda category:1, line_size=10**8):
	'''
		get worddict (need further optimization)
	'''
	stopwords = {line.strip().decode('utf-8') for line in fileinput.input('stopwords.txt')}
	if not os.path.isfile(savefile):
		worddict = defaultdict(int)
		for line in fileinput.input(filename):
			if fileinput.lineno() == line_size: break
			for word in re.split(ur'\s',line.strip().decode('utf-8','ignore')): worddict[word] += 1
		fileinput.close()
		keyworddict = defaultdict(dict)
		get_name_keywords = lambda l:(l[0],l[1:])
		for filename in glob.glob(basedir):
			category = os.path.splitext(os.path.basename(filename))[0].decode('utf-8')
			if not category in categories: continue
			for line in fileinput.input(filename):
				name, keywords = get_name_keywords(re.split(ur'\t',line.strip().decode('utf-8','ignore')))
				for keyword in keywords:
					if all(worddict.get(word,0)>=threshold(category) for word in keyword.lower().split()) and \
						not keyword.lower() in stopwords and len(keyword)>=3 and not keyword.isdigit():
						keyworddict[category][keyword.lower()] = name
			fileinput.close()
			print u'{0}\t{1}'.format(category,len(keyworddict[category]))
		with open(savefile,'w') as outfile: outfile.write(json.dumps(keyworddict))
	else:
		keyworddict = json.loads(open(savefile).read())
	return keyworddict


if __name__ == '__main__':
	start = time.clock()

	# transform_to_tsv()

	# threshold = lambda category: 100 if category in (u'diseases',u'genes') else 10
	# get_worddict(threshold=threshold)
	threshold = lambda category: 30 if category in (u'diseases',u'genes') else 3
	get_worddict(line_size=11000, threshold=threshold)

	elapsed = time.clock()-start
	print '\nTime used:', elapsed

