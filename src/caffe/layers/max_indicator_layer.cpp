#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/max_indicator_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaxIndicatorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void MaxIndicatorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), (Dtype) 0., top_data);
  for (int n = 0; n < num; n++) {
    Dtype max = bottom_data[n*dim];
	int idx = 0;
	for (int i = 1; i < dim; i++) {
	  Dtype val = bottom_data[n*dim + i];
	  if (val > max) {
	    max = val;
		idx = i;
	  }
	}
	top_data[n*dim + idx] = 1;
  }
}

template <typename Dtype>
void MaxIndicatorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // not differentiable
}

#ifdef CPU_ONLY
//STUB_GPU(MaxIndicatorLayer);
#endif

INSTANTIATE_CLASS(MaxIndicatorLayer);
REGISTER_LAYER_CLASS(MaxIndicator);

}  // namespace caffe
