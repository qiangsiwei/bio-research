# -*- coding: utf-8 -*-

'''
	StanfordParser
	StanfordDependencyParser
	StanfordNeuralDependencyParser (corenlp)
'''

'''
	configuration:
	export STANFORD_PARSER_PATH='/Users/kroegera/Documents/_project/bio-research/code_disease/stanford_parser'
	export CLASSPATH='$STANFORD_PARSER_PATH/stanford-parser.jar:$STANFORD_PARSER_PATH/stanford-parser-3.6.0-models.jar:$STANFORD_PARSER_PATH/slf4j-api.jar'
'''

# http://www.zmonster.me/2016/06/08/use-stanford-nlp-package-in-nltk.html

import os
import re
import json
import time
import fileinput
import networkx as nx
from nltk.parse.stanford import StanfordParser, StanfordDependencyParser, StanfordNeuralDependencyParser
from nltk.draw.tree import draw_trees


class RelationExtractor:
	'''
		relation extraction
	'''
	def __init__(self, model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'):
		'''
			initialization
		'''
		self.eng_parser = StanfordParser()
		# self.eng_parser_dependency = StanfordDependencyParser()
		self.eng_parser_dependency = StanfordNeuralDependencyParser()
		
	def PCFG_parse(self, sentence, draw_graph=True):		
		res = list(self.eng_parser.parse(sentence.split()))
		if draw_graph: res[0].draw()
		return res

	def dependency_parse(self, sentence, draw_graph=True):
		res = list(self.eng_parser_dependency.parse(sentence.split()))
		if draw_graph: res[0].tree().draw()
		return res

	def generate_relation(self, sentence, nes, draw_graph=False, dep_path_max=10**2):
		pairs = [(nes[i],nes[j]) for i in range(len(nes)-1) for j in range(i+1,len(nes)) if nes[i][1] != nes[j][1]]
		if len(sentence.split())>60 or len(pairs)<3: return

		def get_relation(n1,n2):
			get_range = lambda n:range(n[2]+1,n[3]+2)
			e1ind, e2ind = get_range(n1), get_range(n2)
			dep_path = nx.shortest_path(G,source=e1ind[-1],target=e2ind[-1])
			vbs = filter(lambda n:G.node[n]['tag'].startswith('VB'),dep_path)
			if len(dep_path) <= dep_path_max and vbs:
				ws = sentence.split(); r = G.node[vbs[-1]]['word']
				e1, e2 = ' '.join(ws[i-1] for i in e1ind), ' '.join(ws[i-1] for i in e2ind)
				print '{0}\n{1} | {2} | {3} | {4}'.format(sentence,e1,e2,r,len(dep_path))
				return e1, e2, r, len(dep_path)
			else:
				return None, None, None, None

		rels = []; res = self.dependency_parse(sentence,draw_graph=False)
		G = nx.Graph(); nodes = {}; edges = []
		return res[0].nodes.items()
		# for key, value in res[0].nodes.iteritems():
		# 	if key: G.add_node(key, word=value['word'], tag=value['tag'])
		# 	if value['head']: G.add_edge(key, value['head'], rel=value['rel'])
		# # for n in range(len(G.node)): print n, G.node.get(n+1)
		# for n1,n2 in pairs:
		# 	e1, e2, r, dep_path_len = get_relation(n1,n2)
		# 	if r: rels.append([n1, n2, r, dep_path_len])
		# return rels


if __name__ == '__main__':
	start = time.clock()

	extractor = RelationExtractor()
	# extractor.dependency_parse('likewise , 5,6-benzoflavone , indole-3-carbinol , 1,2-dithiole-3-thione and oltipraz failed to modulate apoptosis in the respiratory tract of ecs-exposed rats .')

	with open('../data/lung_cancer.rel.tsv','w',0) as outfile:
		for line in fileinput.input('../data/lung_cancer.ner.tsv'):
			print fileinput.lineno()
			try:
				sentence, nes = line.strip().split('\t'); nes = sorted(json.loads(nes),key=lambda x:x[2])
				rels = extractor.generate_relation(sentence, nes)
				# if rels: outfile.write('{0}\t{1}\t{2}\n'.format(sentence,json.dumps(nes),json.dumps(rels)))
				if rels: outfile.write('{0}\t{1}\t{2}\n'.format(sentence,nes,json.dumps(rels)))
			except: continue

	elapsed = time.clock()-start
	print '\nTime used:', elapsed

