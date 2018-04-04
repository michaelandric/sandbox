from elasticsearch import helpers
from elasticsearch import Elasticsearch
import csv

es = Elasticsearch()

with open('/Users/michael/Documents/workspace/docomo_data/subset_data1_1_201607.csv') as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, index='my-index', doc_type='my-type')
