
import os
import sys
import lmdb
import caffe.proto.caffe_pb2
import collections
import math
import json


env = lmdb.open(sys.argv[1], readonly=True, map_size=int(2 ** 42), writemap=False)
txn = env.begin(write=False)
cursor = txn.cursor()
c = collections.Counter()

idx = 0
for key, val in cursor:
	d = caffe.proto.caffe_pb2.DocumentDatum()
	d.ParseFromString(val)	
	print d.dbid
	c[d.dbid] += 1

	idx += 1

env.close()

print c


