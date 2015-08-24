
#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void BinaryEncodeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_bits_ = this->layer_param_.binary_param().num_bits();
  CHECK_GE(num_bits_, 1) << " num_bits must not be less than 1";
}

template <typename Dtype>
void BinaryEncodeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape;
  shape.push_back(bottom[0]->shape(0));
  shape.push_back(num_bits_);
  top[0]->Reshape(shape);
}

template <typename Dtype>
void BinaryEncodeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // number of instances
  int num = bottom[0]->shape(0);
  // number of Dtypes input for each instance
  int count = bottom[0]->count(1);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // number of bytes in a Dtype
  int dtype_size = sizeof(Dtype);
  // number of bits of input for each instance
  int input_bits = 8 * dtype_size * count;
  CHECK_EQ(sizeof(unsigned char), 1);

  for (int n = 0; n < num; n++) {
    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(bottom_data + (n * count));
	unsigned char cur_byte;
  	int cur_bit_idx = 0;
	int cur_bit = 0;
	while (cur_bit_idx < num_bits_ && cur_bit_idx < input_bits) {
	  cur_byte = bytes[cur_bit_idx / 8];
	  cur_bit = (1 << (cur_bit_idx % 8)) & cur_byte;

	  // write a 1 if the cur_bit is on, else a 0
	  if (cur_bit) {
	    top_data[n * count + cur_bit_idx] = 1;
	  } else {
	    top_data[n * count + cur_bit_idx] = 0;
	  }
      cur_bit_idx++;
	}
  }
}

INSTANTIATE_CLASS(BinaryEncodeLayer);
REGISTER_LAYER_CLASS(BinaryEncode);

} // caffe namespace
