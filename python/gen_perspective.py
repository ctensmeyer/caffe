
import random

N = 10
interval=0.00002
cur = interval

for _ in xrange(N):
	i = map(lambda _: random.choice([-1, 1]) * random.uniform(cur, cur+interval),  xrange(8))
	i = map(str, i)
	cur += interval
	print "perspective %s" % " ".join(i)
	
