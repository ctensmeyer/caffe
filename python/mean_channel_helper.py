
import sys

delim=':'

output_type = sys.argv[1]
sources_file = sys.argv[2]
means_file = sys.argv[3]

sources = map(lambda s: s.rstrip(), open(sources_file, 'r').readlines())
means = map(lambda s: int(s.rstrip()), open(means_file, 'r').readlines())
num_sources = len(sources) / 2
mean_idx = 0
mean_strs = list()
channel_strs = list()
try:
	for x in xrange(num_sources):
		source = sources[x]
		if "color" in source:
			source_means = means[mean_idx:mean_idx+3]
			mean_idx += 3
			mean_str = ",".join(map(str, source_means))
			mean_strs.append(mean_str)
			channel_strs.append('3')
		elif ("gray" in source or "binary" in source):
			mean_strs.append(str(means[mean_idx]))
			mean_idx += 1
			channel_strs.append("1")
		else:
			raise Exception("Unrecognized type of source: %s" % source)
except IndexError as e:
	pass
		
if output_type == 'mean':
	print delim.join(mean_strs)
else:
	print delim.join(channel_strs)
		
