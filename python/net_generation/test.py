import caffe
from caffe import layers as L




n = caffe.NetSpec()

n.data = L.Soul()

#n.data = L.Data(source="Test.txt", batch_size=1)
n.layer = L.Convolution(n.data, param=[dict(lr_mult=1), dict(lr_mult=2)], weight_filler=dict(type='msra'))

net = n.to_proto()

for l in net.layer:
    print l.type


net.input.extend(["data"])
net.input_dim.extend([1,3,227,227])

print str(net)
