
import os
import dpp
import caffe
import argparse
import numpy as np
import caffe.proto.caffe_pb2
import google.protobuf.text_format
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn import linear_model



def log(args, message, newline=True):
    '''prints s and optionally logs to a file'''
    print message
    if args.log_file:
        if not hasattr(args, 'log'):
            args.log = open(args.log_file, 'w')
        if newline:
            args.log.write("%s\n" % message)
        else:
            args.log.write(message)


def init_caffe(args):
    '''Loads a caffe model based on the network prototxt and weights'''
    model = caffe.Net(args.network_file, args.weight_file, caffe.TEST)
    return model


def gram(mat, args):
    '''Computes the Gram Matrix of mat according to the kernel specified in
    args'''

    if args.kernel in ['rbf', 'polynomial', 'poly', 'laplacian']:
        gamma = dict(gamma=10. / mat.shape[1])
        output = pairwise_kernels(mat, metric=args.kernel, n_jobs=-1,
            gamma=gamma)
    else:
        # gamma for chi squared should be left to default
        output = pairwise_kernels(mat, metric=args.kernel, n_jobs=-1)
    return output


def sample_neurons(sim_mat, args):
    '''Samples diverse neurons through a Determinental Point Process
    specified by a similarity matrix'''
    return dpp.sample_dpp(sim_mat, args.num_neurons)


def get_layer(layer_name, model):
    '''Returns a layer object based on a layer_name'''
    for layer, name in zip(model.layers, model._layer_names):
        if name == layer_name:
            return layer
    return None


def load_net_spec(args):
    '''Loads the specified network .prototxt file'''
    net_spec = caffe.proto.caffe_pb2.NetParameter()
    _str = open(args.network_file, 'r').read()
    google.protobuf.text_format.Merge(_str, net_spec)
    return net_spec


def save_net_spec(netspec, args):
    '''Saves the network as a .prototxt file'''
    fd = open(args.out_network_file, 'w')
    fd.write(str(netspec))
    fd.close()


def get_layer_param(layer_name, netspec):
    '''Returns a layer_param based on a layer_name'''
    for layer_param in netspec.layer:
        if layer_param.name == layer_name:
            return layer_param
	return None


def update_weights(input_activations, keep, output_activations):
    '''Finds the optimal weighting to map input_activations to
    output_activations given the indices of the neurons to keep'''
    input_activations = input_activations[:, keep]
    log(args, "Pruned input shape: %s" % str(input_activations.shape))
    clf = linear_model.LinearRegression(fit_intercept=True)
    clf.fit(input_activations, output_activations)
    weights = clf.coef_
    intercept = clf.intercept_

    return weights, intercept


def print_model_shape(model):
    '''Prints the shape of the weight and bias matrices for each
    InnerProduct Layer'''
    log(args, "\nModel Shape")
    for layer_name, layer in zip(model._layer_names, model.layers):
        if layer.type == "InnerProduct":
            log(args, "Layer %s: %r %r" % (layer_name,
                layer.blobs[0].data.shape, layer.blobs[1].data.shape))


def main(args):
    '''Performs DivNet pruning'''
    log(args, str(args))

    log(args, "Loading network and weights")
    model = init_caffe(args)
    netspec = load_net_spec(args)
    all_ip_layer_names = [name for (layer, name)
        in zip(model.layers, model._layer_names)
        if layer.type == "InnerProduct"]
    if args.layers == "_all":
        # only include layers with modifiable parameters
        layer_names = list(all_ip_layer_names)
        del layer_names[0]  # cannot prune input neurons
    else:
        layer_names = args.layers.split(args.delimiter)

    log(args, "Layers to prune: %r" % layer_names)
    log(args, "All Layers: %r" % all_ip_layer_names)
    print_model_shape(model)

    for layer_name in layer_names:
        if layer_name == all_ip_layer_names[0]:
            log(args,
                "Skipping Layer %s.  Cannot prune input neurons" % layer_name)
            continue
        if layer_name not in all_ip_layer_names:
            log(args,
                "Skipping Layer %s.  Not an InnerProduct Layer" % layer_name)
            continue

        log(args, "\nStarting Layer %s" % layer_name)

        layer = get_layer(layer_name, model)
        log(args, "Old Weight Shape: %s" % str(layer.blobs[0].data.shape))
        log(args, "Old Bias Shape: %s" % str(layer.blobs[1].data.shape))
        layer_param = get_layer_param(layer_name, netspec)
        if layer_param is None:
            raise Exception("Layer %s does not exist in file %s" % \
                (layer_name, args.network_file))
        bottom_blob_name = layer_param.bottom[0]
        bottom_activations_file = os.path.join(args.activations_dir,
                                                "%s.txt" % bottom_blob_name)
        bottom_activations = np.loadtxt(bottom_activations_file)
        log(args, "Bottom shape: %s" % str(bottom_activations.shape))

        top_blob_name = layer_param.top[0]
        top_activations_file = os.path.join(args.activations_dir,
            "%s.txt" % top_blob_name)
        top_activations = np.loadtxt(top_activations_file)
        log(args, "Top shape: %s" % str(top_activations.shape))

        # row = instance, col = neuron,
        # To get neuron similarity, we transpose
        gram_matrix = gram(bottom_activations.transpose(), args)
        log(args, "L shape: %s" % str(gram_matrix.shape))
        neuron_indices_to_keep = sample_neurons(gram_matrix, args)

        weights, bias = update_weights(bottom_activations,
                                        neuron_indices_to_keep,
                                        top_activations)
        log(args, "New Weight shape: %s" % str(weights.shape))
        log(args, "New Bias shape: %s" % str(bias.shape))

        # reshape weights of current layer
        layer.blobs[0].reshape(weights.shape[0], weights.shape[1])
        layer.blobs[0].data[:] = weights[:]  # cannot do direct assignment
        layer.blobs[1].data[:] = bias[:]

        # reshape weights of previous layer
        cur_ip_layer_idx = all_ip_layer_names.index(layer_name)
        prev_ip_layer_idx = cur_ip_layer_idx -1
        prev_layer_name = all_ip_layer_names[prev_ip_layer_idx]
        prev_layer = get_layer(prev_layer_name, model)
        log(args, "Prev Layer name: %s" % str(prev_layer_name))

        prev_layer_weights = prev_layer.blobs[0].data
        prev_layer_bias = prev_layer.blobs[1].data
        log(args, "Prev Weight shape: %s" % str(prev_layer_weights.shape))
        log(args, "Prev Bias shape: %s" % str(prev_layer_bias.shape))
        new_prev_layer_weights = prev_layer_weights[neuron_indices_to_keep, :]
        new_prev_layer_bias = prev_layer_bias[neuron_indices_to_keep]
        log(args, "New Prev Weight shape: %s" % str(
            new_prev_layer_weights.shape))
        log(args, "New Prev Bias shape: %s" % str(new_prev_layer_bias.shape))

        prev_layer.blobs[0].reshape(new_prev_layer_weights.shape[0],
                                    new_prev_layer_weights.shape[1])
        prev_layer.blobs[0].data[:] = new_prev_layer_weights[:]
        prev_layer.blobs[1].reshape(new_prev_layer_bias.shape[0])
        prev_layer.blobs[1].data[:] = new_prev_layer_bias[:]

        # modify num_output for previous layer_param
        prev_layer_param = get_layer_param(prev_layer_name, netspec)
        prev_layer_param.inner_product_param.num_output = (
            neuron_indices_to_keep.shape[0])

    save_net_spec(netspec, args)
    print_model_shape(model)
    model.save(args.out_weight_file)


def get_args():
    '''Parses cmd line arguments'''
    parser = argparse.ArgumentParser(description="Classifies data")
    parser.add_argument("network_file",
                help="The model definition file (e.g. deploy.prototxt)")
    parser.add_argument("weight_file",
                help="The model weight file (e.g. net.caffemodel)")
    parser.add_argument("activations_dir",
                help="directory of activation values for each layer")
    parser.add_argument("out_network_file",
                help="Outfile for the pruned caffe architecture")
    parser.add_argument("out_weight_file",
                help="The output model weight file for the pruned caffe model")

    parser.add_argument("-k", "--num-neurons", type=int, default=0,
                help="Specify the number of neurons to keep at each layer. \
                    Leave at 0 to get a variable sized sample from the DPP")
    parser.add_argument("-l", "--layers", type=str, default="_all",
                help="delimited list of layer names to prune")
    parser.add_argument("-d", "--delimiter", default=',', type=str,
                help="Delimiter used for parsing list of layers")
    parser.add_argument("-f", "--log-file", type=str, default="",
                help="Log File")
    parser.add_argument("--kernel", type=str, default='linear',
                help="Kernel used to construct Gram Matrix")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)

