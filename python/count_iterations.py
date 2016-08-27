
import os
import sys
import lmdb

lmdb_dir = sys.argv[1]
batch_size = int(sys.argv[2])

iters = 0
extras = 0
total_size = 0
for db in os.listdir(lmdb_dir):
	try:
		env = lmdb.Environment(os.path.join(lmdb_dir, db), create=False)
	except:
		print "Could not open %s, skipping" % db
	size = env.stat()['entries']
	db_iters = size / batch_size
	extra = size % batch_size
	if extra > 0:
		db_iters += 1
	iters += db_iters
	extras += extra
	total_size += size

	print "DB: %s has %d entries and needs %d iterations (%d on last iter)" % (db, size, db_iters, extra)

print "\n%d Iterations needed total" % iters
print "\n%d In last part total" % extras
print "\n%d Total size" % total_size
	


