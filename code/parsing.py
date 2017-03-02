# -*- encoding:utf-8 -*-


import time
import spacy

# Difference between constituency parser and dependency parser.
# http://stackoverflow.com/questions/10401076/difference-between-constituency-parser-and-dependency-parser

'''
	Constituency tree
	(ROOT
	  (S
	    (NP (NNP John))
	    (VP (VBZ likes)
	      (NP (PRP him)))
	    (. .)))

	Dependency tree
	nsubj(likes-2, John-1)
	root(ROOT-0, likes-2)
	dobj(likes-2, him-3)
'''

def dependency_parsing(sentence, nlp=spacy.load('en')):
	'''
		dependency parsing
	'''
	doc = nlp(sentence)
	for np in doc.noun_chunks:
	    print(np.text, np.root.text, np.root.dep_, np.root.head.text)


if __name__ == '__main__':
	start = time.clock()

	sentence = u'I like green eggs and ham.'
	dependency_parsing(sentence)

	elapsed = time.clock()-start
	print '\nTime used:', elapsed

