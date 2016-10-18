
import random

N = 20
interval=0.0000075
cur = interval

for _ in xrange(N):
	i = map(lambda _: random.choice([-1, 1]) * random.uniform(cur, cur+interval),  xrange(8))
	i = map(str, i)
	cur += interval
	print "perspective %s" % " ".join(i)
	
