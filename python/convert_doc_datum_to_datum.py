
import sys
import lmdb
import StringIO
import caffe.proto.caffe_pb2
import numpy as np
from PIL import Image

def convert(doc_datum):
	datum = caffe.proto.caffe_pb2.Datum()
	datum_im = doc_datum.image

	datum.channels = datum_im.channels
	datum.width = datum_im.width
	datum.height = datum_im.height
	datum.encoded = datum_im.encoding != 'none'
	datum.data = datum_im.data

	#print "%dx%dx%d with %d bytes of data with encoding: %s" % (
	#	datum.channels, datum.height, datum.width, len(datum.data), datum.encoded)

	datum.label = doc_datum.collection
	return datum


def main(in_db, out_db):
	in_env = lmdb.open(in_db, readonly=True, map_size=int(2 ** 42), writemap=False)
	in_txn = in_env.begin(write=False)
	in_cursor = in_txn.cursor()

	out_env = lmdb.open(out_db, readonly=False, map_size=int(2 ** 42), writemap=True)
	out_txn = out_env.begin(write=True)
	for key, value in in_cursor:
		doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
		doc_datum.ParseFromString(value)	
		datum = convert(doc_datum)
		out_txn.put(key, datum.SerializeToString())
		print key

	out_txn.commit()
	out_env.sync()
	out_env.close()
	
	

if __name__ == "__main__":
	in_db = sys.argv[1]
	out_db = sys.argv[2]
	main(in_db, out_db)

