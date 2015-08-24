
import sys
import lmdb
import StringIO
import caffe.proto.caffe_pb2
import numpy as np
from PIL import Image


def main(in_db):
	in_env = lmdb.open(in_db, readonly=True, map_size=int(2 ** 42), writemap=False)
	in_txn = in_env.begin(write=False)
	in_cursor = in_txn.cursor()

	_max = -1
	for key, value in in_cursor:
		dd = caffe.proto.caffe_pb2.DocumentDatum()
		dd.ParseFromString(value)	
		if dd.num_str:
			_max = max(_max, dd.num)

	print _max
	

if __name__ == "__main__":
	in_db = sys.argv[1]
	main(in_db)

