
import os
import sys
import lmdb
import caffe.proto.caffe_pb2
import collections
import math
import json

def entropy(c):
	total = float(sum(c.values()))
	entropy = 0
	for val in c.values():
		prob = val / total
		entropy += -1 * prob * math.log(prob) / math.log(2)
	return entropy

_epsilon = 1e-4
def kl_divergence(c1, c2):
	total1 = float(sum(c1.values()))
	total2 = float(sum(c2.values()))
	divergence = 0
	for key in (set(c1) | set(c2)):
		p = c1[key] / total1 if key in c1 else _epsilon
		q = c2[key] / total2 if key in c2 else _epsilon
		divergence += p * math.log(p / q)
	return divergence

def stats(l):
	mean = sum(l) / float(len(l)) if len(l) else None
	std_dev = math.sqrt(sum(map(lambda x: (x - mean) ** 2, l)) / len(l)) if len(l) else None
	return mean, std_dev
	

in_dir = sys.argv[1]

discrete = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
continuous = collections.defaultdict(lambda: collections.defaultdict(list))

entropies = collections.defaultdict(dict)
divergence = collections.defaultdict(dict)
mean = collections.defaultdict(dict)
std_dev = collections.defaultdict(dict)

dbs = list()
discrete_fields = ['country', 'language', 'is_document', 'is_graphical_document', 
	'is_historical_document', 'is_textual_document', 'collection', 'record_type_fine',
	'record_type_broad', 'layout_type', 'layout_category', 'media_type']
continuous_fields = ['decade', 'column_count', 'possible_records', 'actual_records',
	'pages_per_image', 'docs_per_image', 'machine_text', 'hand_text']

for db in os.listdir(in_dir):
	try:
		env = lmdb.open(os.path.join(in_dir, db), readonly=True, map_size=int(2 ** 42), writemap=False)
		txn = env.begin(write=False)
		cursor = txn.cursor()
	except Exception as e:
		print e
		print "Skipping %s, could not open" % db
		continue
	dbs.append(db)
	
	for key, val in cursor:
		d = caffe.proto.caffe_pb2.DocumentDatum()
		d.ParseFromString(val)	

		if d.HasField('country'):
			discrete[db]['country'][d.country] += 1
		if d.HasField('language'):
			discrete[db]['langauge'][d.language] += 1
		if d.HasField('is_document'):
			discrete[db]['is_document'][d.is_document] += 1
		if d.HasField('is_graphical_document'):
			discrete[db]['is_graphical_document'][d.is_graphical_document] += 1
		if d.HasField('is_historical_document'):
			discrete[db]['is_historical_document'][d.is_historical_document] += 1
		if d.HasField('is_textual_document'):
			discrete[db]['is_textual_document'][d.is_textual_document] += 1
		if d.HasField('collection'):
			discrete[db]['collection'][d.collection] += 1
		if d.HasField('record_type_fine'):
			discrete[db]['record_type_fine'][d.record_type_fine] += 1
		if d.HasField('record_type_broad'):
			discrete[db]['record_type_broad'][d.record_type_broad] += 1
		if d.HasField('layout_type'):
			discrete[db]['layout_type'][d.layout_type] += 1
		if d.HasField('layout_category'):
			discrete[db]['layout_category'][d.layout_category] += 1
		if d.HasField('media_type'):
			discrete[db]['media_type'][d.media_type] += 1
			
		if d.HasField('decade'):
			continuous[db]['decade'].append(d.decade)
		if d.HasField('column_count'):
			continuous[db]['column_count'].append(d.column_count)
		if d.HasField('possible_records'):
			continuous[db]['possible_records'].append(d.possible_records)
		if d.HasField('actual_records'):
			continuous[db]['actual_records'].append(d.actual_records)
		if d.HasField('pages_per_image'):
			continuous[db]['pages_per_image'].append(d.pages_per_image)
		if d.HasField('docs_per_image'):
			continuous[db]['docs_per_image'].append(d.docs_per_image)
		if d.HasField('machine_text'):
			continuous[db]['machine_text'].append(d.machine_text)
		if d.HasField('hand_text'):
			continuous[db]['hand_text'].append(d.hand_text)
			
for disc_field in discrete_fields:
	c = collections.Counter()
	for db in dbs:
		c.update(discrete[db][disc_field])
	discrete['all'][disc_field] = c
	entropies['all'][disc_field] = entropy(c)
	divergence['all'][disc_field] = kl_divergence(c, c)

for cont_field in continuous_fields:
	l = list()
	for db in dbs:
		l += continuous[db][cont_field]
	continuous['all'][cont_field] = l
	m, std = stats(l)
	mean['all'][cont_field] = m
	std_dev['all'][cont_field] = std


for db in dbs:
	for disc_field in discrete_fields:
		c_all = discrete['all'][disc_field]
		c = discrete[db][disc_field]
		ent = entropy(c)
		div = kl_divergence(c, c_all)
		entropies[db][disc_field] = ent
		divergence[db][disc_field] = div
	
	for cont_field in continuous_fields:
		c = continuous[db][cont_field]
		m, s = stats(c)
		mean[db][cont_field] = m
		std_dev[db][cont_field] = s
		
print "Entropy"
print json.dumps(entropies, indent=4)

print "Divergence"
print json.dumps(divergence, indent=4)
		
print "Mean"
print json.dumps(mean, indent=4)
		
print "Std_dev"
print json.dumps(std_dev, indent=4)



