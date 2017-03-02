# -*- encoding:utf-8 -*-

import json
import time
import pymongo
import requests

from config import *
from constant import *


def request_server(url, check_field='', params={}, headers=headers, retry=10):
	for _ in range(retry):
		try:
			r = requests.get(url,params=params,headers=headers)
			if not check_field or check_field in r.json(): return r
		except: pass
	print 'request server error, exceed maximum retry.'; exit()


def get_articles(search_str, years=range(2010,2016), check_field='search-results'):
	key = 'dc:identifier'
	url = 'http://api.elsevier.com/content/search/scopus'
	for year in years:
		for month in range(1,13):
			start, count, total = 0, 25, 5000
			date_begin, date_end = '{0}{1}01'.format(year,str(month).zfill(2)), '{0}{1}31'.format(year,str(month).zfill(2))
			params = {'view':'COMPLETE','query':'abs({0}) AND (Orig-Load-Date AFT {1} AND Orig-Load-Date BEF {2})'.format(search_str,date_begin,date_end)}
			while start < total:
				params.update({'start':start})
				r = request_server(url,check_field=check_field,params=params)
				total = min(total,int(r.json().get(check_field,{}).get('opensearch:totalResults',-1)))
				for entry in r.json().get(check_field,{}).get('entry',[]):
					if key in entry: 
						entry = json.loads(json.dumps(entry).replace('$','d'))
						db['articles'].update({key:entry.get(key,'')},entry,upsert=True)
				print '{0}\t{1}/{2}'.format(date_begin,start,total)
				start += count


if __name__ == '__main__':
	start = time.clock()

	get_articles(search_str='lung AND cancer')

	elapsed = time.clock()-start
	print '\nTime used:', elapsed

