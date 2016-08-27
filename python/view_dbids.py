
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

idx = 0
for key, val in cursor:
	d = caffe.proto.caffe_pb2.DocumentDatum()
	d.ParseFromString(val)	
	print d.dbid

	idx += 1
	if idx > 100:
		break

env.close()



