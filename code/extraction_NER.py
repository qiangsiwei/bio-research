# -*- encoding:utf-8 -*-

import os
import re
import json
import time
import random
import struct
import fileinput
import numpy as np
from collections import *
from nltk.tokenize import sent_tokenize
from keras.models import Graph, model_from_json
from keras.layers import recurrent, Dropout, TimeDistributedDense

from config import *
from constant import *


class WordEmbdedding:
	'''
		get word embdeddings
	'''
	def __init__(self, filename, vec_dim):
		self.vec_dim = vec_dim
		self.__word2embedding = {}
		self.__load_data(filename)
	
	def __load_data(self, filename):
		with open(filename, 'rb') as f:
			data = f.read(self.vec_dim)
			sep = data.find('\n')
			word_n_size = data[:sep]
			words, size = word_n_size.split()
			words, size = int(words), int(size)
			data = data[sep+1:]
			for b in range(words):
				c_data = f.read(self.vec_dim+1)
				data = data + c_data
				separator = data.find(' ')
				word = data[:separator].decode('utf-8')
				data = data[separator+1:]
				if len(data) < 4*size:  # assuming 4 byte float
					data += f.read(4*size)
				vec = np.array(struct.unpack('{0}f'.format(size), data[:4*size]))
				length = np.sqrt((vec**2).sum())
				vec /= length
				data = data[4*size+1:]
				self.__word2embedding[word.lower()] = vec

	def has(self, word):
		return word.lower() in self.__word2embedding

	def get(self, word):
		return self.__word2embedding.get(word.lower(), [0]*self.vec_dim)


class EntityTagger:
	'''
		entity extraction
	'''
	def __init__(self, embeddingfilename='../data/word2vec/vectors.bin', vec_dim=200, worddictfilename='worddict.json'):
		'''
			initialization
		'''
		self.vec_dim = vec_dim
		self.embeddings = WordEmbdedding(embeddingfilename, vec_dim)
		self.categories = (u'diseases',u'genes',u'drugs',u'hmdb',u'organs',u'tissues',u'adverse_effects',u'drug_effects') # categories
		self.worddict = json.loads(open(worddictfilename).read())
		self.regexlist = [(category,re.compile(ur'(?<=\s)({0})(?=\s)'.format(re.sub(ur'[\[\]\{\}\,\(\)\+\*\?\^\$]',u'\\\1',u'|'.join(sorted(self.worddict[category].keys(),key=len,reverse=True)))),flags=re.I)) for category in self.categories]
		self.stopwords = {line.strip().decode('utf-8') for line in fileinput.input('stopwords.txt')}
		self.KB = {i+1:defaultdict(int) for i,category in enumerate(self.categories)}

	def __generate_segments(self, ls):
		'''
			generate segments
		'''
		y2 = np.zeros((2,len(ls)))
		for i in xrange(len(ls)):
			if i >= 1 and ls[i-1] and ls[i] == ls[i-1]: y2[1][i] = 1
		y2 = y2.T; y2[:,0] = y2.sum(axis=1) == 0
		return y2

	def __generate_phrases(self, ls, ws, cs=None, return_array=False):
		'''
			generate phrases for results
		'''
		if isinstance(cs, np.ndarray):
			for i in range(1,len(ls)): 
				if cs[i] and not ls[i]: ls[i] = ls[i-1]
			for i in range(len(ls)-1,0,-1): 
				if cs[i] and not ls[i-1]: ls[i-1] = ls[i]
		phrases = {}; current = None
		for i in xrange(len(ls)):
			if ls[i] == 0 or i >= 1 and ls[i-1] and ls[i] != ls[i-1]:
				if current: phrases[tuple(current[2:])] = current; current = None
			else:
				if not current: current = [ws[i], ls[i], i, i]; continue
				current[0] += u' '+ws[i]; current[-1] = i
		else: 
			if current: phrases[tuple(current[2:])] = current
		if return_array: return phrases, ls, cs
		return phrases

	def __generate_lables(self, line):
		'''
			generate lables for training and testing
		'''
		line = re.sub(ur'\s+',u' ',line.strip().decode('utf-8'))
		x = map(self.embeddings.get, line.split()); y1 = [[0]*len(x)]
		for category, regex in self.regexlist:
			word_pos = reduce(lambda x,y: x+[(x[-1][-1]+1,x[-1][-1]+1+len(y))], line.split(), [(-1,)])[1:]
			regex_pos = [(mobj.start(),mobj.end()) for mobj in regex.finditer(line)]
			y1.append([1 if any(rb <= wb < rf or rb < wf <= rf for rb, rf in regex_pos) else 0 for wb, wf in word_pos])
		x, y1 = np.array(x), np.array(y1).T; y1[:,0] = y1.sum(axis=1) == 0; y2 = self.__generate_segments(y1.argmax(axis=1))
		phrases = self.__generate_phrases(y1.argmax(axis=1), line.split())
		return line, x, y1, y2, phrases

	def __update_KB(self, l, x, q1, q2, show_text=False, threshold=1):
		'''
			update entity knowledge base
		'''
		last_i, last_z, last_w = -1, -1, ''
		for (i,w),z,c in zip(enumerate(l.split()),q1,q2):
			if z == last_z and i == last_i+1:
				last_w = last_w+' '+w if last_w else w; last_i = i
			else:
				if len(last_w) >= 2 and not last_w.isdigit() and not any(w in self.stopwords for w in last_w.split()):
					if show_text: print '\033[41;37m{0}:{1}\033[0m'.format(last_z,last_w),
					if not last_w in self.worddict[self.categories[last_z-1]]: self.KB[last_z][last_w] += 1
					last_i, last_z, last_w = -1, -1, ''
					# self.KB[last_z][last_w] += 1; last_i, last_z, last_w = -1, -1, ''
				if z: last_i, last_z, last_w = i, z, w
			if show_text: print '{0}:{1}:{2}'.format(z,c,w),

	def __output_KB(self, savefilename):
		'''
			output entity knowledge base
		'''
		with open(savefilename,'w') as outfile:
			for z, wdict in self.KB.iteritems():
				for w, count in sorted(wdict.iteritems(),key=lambda x:x[1],reverse=True):
					try:
						if count >= 1: outfile.write(u'{0}\t{1}\t{2}\n'.format(self.categories[z-1],w,count).encode('utf-8'))
					except: continue

	def train(self, trainingfilename, train_size=2000, test_size=1000, mode='single_task', RNN=recurrent.GRU, HIDDEN_SIZE=256, epoch_size=5, epoch_update=2, evaluate=False, base_dir='../data/models'):
		'''
			main function for training
		'''
		# build model
		print 'compile model'
		model = Graph()
		if mode == 'single_task':
			model.add_input(name='input', input_shape=(None, self.vec_dim))
			model.add_node(RNN(HIDDEN_SIZE, activation='relu', return_sequences=True), name='forward', input='input')
			model.add_node(RNN(HIDDEN_SIZE, activation='relu', return_sequences=True, go_backwards=True), name='backward', input='input')
			model.add_node(Dropout(0.5), name='dropout', merge_mode='concat', inputs=['forward', 'backward'])
			model.add_node(TimeDistributedDense(len(self.categories)+1, activation='softmax'), name='softmax', input='dropout')
			model.add_output(name='output1', input='softmax')
			model.compile('adam', {'output1': 'categorical_crossentropy'})

		elif mode == 'uniform_layer':
			model.add_input(name='input', input_shape=(None, self.vec_dim))
			model.add_node(RNN(HIDDEN_SIZE, activation='relu', return_sequences=True), name='forward', input='input')
			model.add_node(RNN(HIDDEN_SIZE, activation='relu', return_sequences=True, go_backwards=True), name='backward', input='input')
			model.add_node(Dropout(0.5), name='dropout', merge_mode='concat', inputs=['forward', 'backward'])
			model.add_node(TimeDistributedDense(len(self.categories)+1, activation='softmax'), name='softmax1', input='dropout')
			model.add_node(TimeDistributedDense(2, activation='softmax'), name='softmax2', input='dropout')
			model.add_output(name='output1', input='softmax1')
			model.add_output(name='output2', input='softmax2')
			model.compile(optimizer='adam', loss={'output1':'categorical_crossentropy','output2':'categorical_crossentropy'}, loss_weights={'output1':0.5,'output2':0.5})
		
		elif mode == 'coupled_layer':
			model.add_input(name='input', input_shape=(None, self.vec_dim))
			model.add_node(RNN(HIDDEN_SIZE, activation='relu', return_sequences=True), name='forward', input='input')
			model.add_node(RNN(HIDDEN_SIZE, activation='relu', return_sequences=True, go_backwards=True), name='backward', input='input')
			model.add_node(Dropout(0.3), name='dropout', merge_mode='concat', inputs=['forward', 'backward'])
			# model.add_node(TimeDistributedDense(len(self.categories)+1, activation='softmax'), name='softmax1', input='dropout')
			# model.add_output(name='output1', input='softmax1')
			model.add_node(TimeDistributedDense(2, activation='softmax'), name='softmax2', input='dropout')
			model.add_output(name='output2', input='softmax2')
			model.add_node(RNN(HIDDEN_SIZE, activation='relu', return_sequences=True), name='forward2', input='dropout')
			model.add_node(RNN(HIDDEN_SIZE, activation='relu', return_sequences=True, go_backwards=True), name='backward2', input='dropout')
			model.add_node(Dropout(0.3), name='dropout2', merge_mode='concat', inputs=['forward2', 'backward2'])
			# model.add_node(TimeDistributedDense(2, activation='softmax'), name='softmax2', input='dropout2')
			# model.add_output(name='output2', input='softmax2')
			model.add_node(TimeDistributedDense(len(self.categories)+1, activation='softmax'), name='softmax1', input='dropout2')
			model.add_output(name='output1', input='softmax1')
			model.compile(optimizer='adam', loss={'output1':'categorical_crossentropy','output2':'categorical_crossentropy'}, loss_weights={'output1':0.6,'output2':0.4})
		else:
			raise Exception('Mode not supported.')
		with open(os.path.join(base_dir,'model.json'),'w') as model_file: model_file.write(model.to_json())

		# load data
		data = open(trainingfilename).readlines()[:train_size+test_size]
		train_data, test_data = data[:train_size], data[-test_size:]

		# train model
		print 'start training'
		for epoch in xrange(epoch_size):
			for lineno, l in enumerate(train_data):
				if lineno % 10**2 == 0: print 'Train\tepoch:{0}\tlineno:{1}'.format(epoch,lineno)
				# read
				_, x, y1, y2, _ = self.__generate_lables(l)
				if mode == 'single_task':
					model.train_on_batch({'input':np.array([x]), 'output1':np.array([y1])})
				else:
					model.train_on_batch({'input':np.array([x]), 'output1':np.array([y1]), 'output2':np.array([y2])})
				# write
				if epoch >= epoch_update:
					q = model.predict({'input':np.array([x])})
					self.__update_KB(l,x,q['output1'][0].argmax(axis=1),q['output2'][0].argmax(axis=1))
				if (lineno+1) % 2500 == 0:
					# evaluate model
					if evaluate: self.__do_evaluate(model, test_data, base_dir)
					# save model parameters
					model.save_weights(os.path.join(base_dir,'weights_epoch{0}.{1}.h5'.format(epoch,lineno)), overwrite=True)
					# self.__output_KB(os.path.join(base_dir,'KB_dump{0}.tsv'.format(epoch)))

	def __do_evaluate(self, model, test_data, base_dir):
		'''
			do evaluate
		'''
		correct, incorrect, labeled, predicted = [0]*2, [0]*2, [0]*2, [0]*2
		p_correct, p_labeled, p_predicted = 0, 0, 0
		for lineno, l in enumerate(test_data):
			if lineno % 50 == 0: print 'Test\tlineno:{0}'.format(lineno)
			line, x, y1, y2, phrases = self.__generate_lables(l)
			for ind, name, y in [(0,'output1',y1),(1,'output2',y2)]:
				p = np.array(y).argmax(axis=1)
				q = model.predict({'input':np.array([x])})[name][0].argmax(axis=1)
				correct[ind] += ((p==q)*(p!=0)).sum(); incorrect[ind] += (p!=q).sum()
				labeled[ind] += (p!=0).sum(); predicted[ind] += (q!=0).sum()
				if name == 'output1':
					phrases_p = self.__generate_phrases(p, line.split())
					q = model.predict({'input':np.array([x])})
					phrases_q, q1, q2 = self.__generate_phrases(q['output1'][0].argmax(axis=1), line.split(), q['output2'][0].argmax(axis=1), return_array=True)
					p_correct += len(set(phrases_p.keys())&set(phrases_q.keys()))
					p_labeled += len(phrases_p.keys()); p_predicted += len(phrases_q.keys())
					self.__update_KB(l,x,q1,q2,show_text=False)
		print "\tCorrect:{0}\tIncorrect:{1}\tLabeled:{2}\tPredicted:{3}".format(correct, incorrect, labeled, predicted)		
		print "\tPhrase_Correct:{0}\tPhrase_Labeled:{1}\tPhrase_Predicted:{2}\tElapsed:{3}".format(p_correct, p_labeled, p_predicted, time.clock()-start)
		# self.__output_KB(os.path.join(base_dir,'KB_dump.evaluation.tsv'))

	def evaluate(self, trainingfilename, line_start=1000, line_size=100, base_dir='../data/models', iteration=0):
		'''
			main function for evaluation
		'''
		# load data
		test_data = open(trainingfilename).readlines()[line_start:line_start+line_size]
		# load model
		modelfilename = os.path.join(base_dir,'model.json')
		paramfilename = os.path.join(base_dir,'weights_epoch{0}.h5'.format(iteration))
		model = model_from_json(open(modelfilename,'r').read())
		model.load_weights(paramfilename)
		# evaluate model
		self.__do_evaluate(model, test_data, base_dir)

	def __do_tag(self, model, data, base_dir='../data', outfilename='lung_cancer.ner.tsv'):
		'''
			do tag
		'''
		with open(os.path.join(base_dir,outfilename),'w') as outfile:
			for lineno, l in enumerate(data):
				if lineno % 50 == 0: print 'Tag\tlineno:{0}'.format(lineno)
				title, abstracts = map(lambda x:x.decode('utf-8'),l.split('\t'*3)); sentences = sent_tokenize(abstracts)
				outfile.write(u'{0}\n'.format(title).encode('utf-8'))
				for sentence in sentences:
					try:
						sentence = re.sub(ur'\s+',u' ',sentence).strip()
						x = map(self.embeddings.get, sentence.split())
						q = model.predict({'input':np.array([x])})
						phrases = self.__generate_phrases(q['output1'][0].argmax(axis=1), sentence.split(), q['output2'][0].argmax(axis=1))
						outfile.write(u'{0}\t{1}\n'.format(sentence,json.dumps(phrases.values())).encode('utf-8'))
					except: continue

	def tag(self, filename, line_size=1000, base_dir='../data/models', iteration=0):
		'''
			main function for evaluation
		'''
		# load data
		data = open(filename).readlines()[:line_size]
		# load model
		modelfilename = os.path.join(base_dir,'model.json')
		paramfilename = os.path.join(base_dir,'weights_epoch{0}.h5'.format(iteration))
		model = model_from_json(open(modelfilename,'r').read())
		model.load_weights(paramfilename)
		# evaluate model
		self.__do_tag(model, data)


if __name__ == '__main__':
	start = time.clock()

	tagger = EntityTagger()
	# tagger.train('../data/lung_cancer.tsv', train_size=800, test_size=200, mode='single_task', epoch_size=5, epoch_update=0, evaluate=True)
	# tagger.train('../data/lung_cancer.tsv', train_size=800, test_size=200, mode='uniform_layer', epoch_size=8, epoch_update=0, evaluate=True)
	# tagger.train('../data/lung_cancer.tsv', train_size=10000, test_size=1000, mode='coupled_layer', epoch_size=20, epoch_update=0, evaluate=True)
	# tagger.evaluate('../data/lung_cancer.tsv', line_start=10000, line_size=1000, base_dir='../data/models', iteration='8.7499')
	tagger.tag('../data/lung_cancer.tsv', line_size=60000, base_dir='../data/models', iteration='8.7499')

	elapsed = time.clock()-start
	print '\nTime used:', elapsed

