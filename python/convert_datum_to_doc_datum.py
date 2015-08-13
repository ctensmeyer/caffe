
import sys
import lmdb
import StringIO
import caffe.proto.caffe_pb2
import numpy as np
from PIL import Image

def convert(datum):
	doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
	datum_im = doc_datum.image

	datum_im.channels = datum.channels
	datum_im.width = datum.width
	datum_im.height = datum.height
	datum_im.encoding = "unknown?" if datum.encoded else "none"
	datum_im.data = datum.data

	#print "%dx%dx%d with %d bytes of data with encoding: %s" % (
	#	datum.channels, datum.height, datum.width, len(datum.data), datum.encoded)

	doc_datum.collection = datum.label
	return doc_datum


def main(in_db, out_db):
	in_env = lmdb.open(in_db, readonly=True, map_size=int(2 ** 42), writemap=False)
	in_txn = in_env.begin(write=False)
	in_cursor = in_txn.cursor()

	out_env = lmdb.open(out_db, readonly=False, map_size=int(2 ** 42), writemap=True)
	out_txn = out_env.begin(write=True)
	for key, value in in_cursor:
		datum = caffe.proto.caffe_pb2.Datum()
		datum.ParseFromString(value)	
		doc_datum = convert(datum)
		out_txn.put(key, doc_datum.SerializeToString())
		print key

	out_txn.commit()
	out_env.sync()
	out_env.close()
	
	

if __name__ == "__main__":
	in_db = sys.argv[1]
	out_db = sys.argv[2]
	main(in_db, out_db)

