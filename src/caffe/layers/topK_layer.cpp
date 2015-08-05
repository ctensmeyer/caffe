#include <algorithm>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Container>
struct compare_indirect_index_ascend {
  const Container& container;
  explicit compare_indirect_index_ascend(const Container& container) :
    container(container) {}

  bool operator()(size_t lindex, size_t rindex) const {
    return container[lindex] < container[rindex];
    }
};

template <typename Container>
struct compare_indirect_index_descend {
  const Container& container;
  explicit compare_indirect_index_descend(const Container& container) :
    container(container)  {}

  bool operator()(size_t lindex, size_t rindex) const {
    return container[lindex] > container[rindex];
    }
};

#if __cplusplus > 199711L
template <typename Dtype>
void kth_element_idxs(const std::vector<Dtype> &v,
  std::vector<size_t> &idx, const size_t k, const int sortAscend = 1) {
  if (sortAscend) {
      std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                compare_indirect_index_ascend <std::vector<Dtype> > (v));
    } else {
      std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                compare_indirect_index_descend <std::vector<Dtype> > (v));
    }
  return;
}
#else

template <typename Dtype>
void sort_idxs(const std::vector<Dtype> &v, std::vector<size_t> &idx,
                                 const int sortAscend = 1) {
  if (sortAscend) {
      std::sort(idx.begin(), idx.end(),
                compare_indirect_index_ascend <std::vector<Dtype> > (v));
  } else {
      std::sort(idx.begin(), idx.end(),
                compare_indirect_index_descend <std::vector<Dtype> > (v));
  }
  return;
}
#endif

// wrapper for impl based on availability of c++11 std::nth_element function.
// At a bare minimum, the kth element of $idx will be in its sorted position,
// and all elements preceeding the kth will be less than or greater the kth 
// depending on sortAscend.  The elements of $idx are indices into $v, and the
// sort comparator will compare positions i, j as v[idx[i]] < v[idx[j]]
// When c++11 is not available, then $idx is sorted completely, fulfilling
// the purpose of this function + more.
template <typename Dtype>
void TopKLayer<Dtype>::kth_element_idxs_(const std::vector<Dtype> &v, std::vector<size_t> &idx,
                                 const size_t k, const int sortAscend) {
#if __cplusplus > 199711L
	kth_element_idxs(v, idx, k, sortAscend);
#else
	sort_idxs(v, idx, sortAscend);
#endif
}

template <typename Dtype>
void TopKLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  mask_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void TopKLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  TopKParameter topk_param = this->layer_param_.topk_param();

  CHECK(topk_param.has_k_num() != topk_param.has_k_perc()) << "Cannot specify both k_num and k_perc";
  if (topk_param.has_k_num()) {
    CHECK_GT(topk_param.k_num(), 0) << "Must be positive";
    k_ = topk_param.k_num();
	k_perc_ = 0;
	k_is_perc_ = false;
  } else {
    CHECK_GT(topk_param.k_perc(), 0) << "Must be positive";
    CHECK_LE(topk_param.k_perc(), 1) << "Must be less than or equal 1";
    k_perc_ = topk_param.k_perc();
	k_ = 0;
	k_is_perc_ = true;
  }
  a_ = topk_param.a();
  CHECK_GT(a_, 0) << "TEST phase multiplier must be positive";
  grouping_ = topk_param.grouping();

  CHECK_GE(bottom[0]->num_axes(), 2) << "Must have at least 2 axes";

  if (grouping_ != TopKParameter::LATERAL && grouping_ != TopKParameter::LIFETIME) {
    CHECK(bottom[0]->num_axes() == 4) << "CONV_* groupings require bottom blobs with 4 axes";
  }
}

template <typename Dtype>
void TopKLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
	switch (grouping_) {
	  case TopKParameter::LATERAL:
	    Forward_cpu_lateral(bottom[0], top[0]);
		break;
	  case TopKParameter::LIFETIME:
	    Forward_cpu_lifetime(bottom[0], top[0]);
		break;
	  case TopKParameter::CONV_SPATIAL:
	    Forward_cpu_conv_spatial(bottom[0], top[0]);
		break;
	  case TopKParameter::CONV_CROSS_CHANNEL:
	    Forward_cpu_conv_cross_channel(bottom[0], top[0]);
		break;
	  case TopKParameter::CONV_LIFETIME_SPATIAL:
	    Forward_cpu_lifetime(bottom[0], top[0]);
		break;
	  case TopKParameter::CONV_LIFETIME_CHANNEL:
	    Forward_cpu_conv_lifetime_channel(bottom[0], top[0]);
		break;
	  default:
	    CHECK(0) << "Unrecognized TopKParameter::Grouping " << grouping_;
	}
}

template <typename Dtype>
int TopKLayer<Dtype>::compute_k(const int num_per_slice) {
  float k = 0;
  if (k_is_perc_) {
    k = k_perc_ * num_per_slice;
  } else {
    k = k_;
  }
  if (this->phase_ == TEST) {
    k *= a_;
  }
  // truncate k if too large or small
  if (k < 1) {
    k = 1;
  }
  return k < num_per_slice ? (int) k : num_per_slice;
}

template <typename Dtype>
void TopKLayer<Dtype>::Forward_cpu_lateral(const Blob<Dtype>* bottom,
                                     Blob<Dtype>* top) {
	CHECK_GE(bottom->num_axes(), 2) << "Must have dimensionality of at least 2";
	DCHECK(top->shape() == bottom->shape()) << "Top and bottom must have same shape";

    const Dtype* bottom_data = bottom->cpu_data();
    Dtype* top_data = top->mutable_cpu_data();
    uint* mask = mask_.mutable_cpu_data();
	const int count = bottom->count();

    caffe_set(count, Dtype(0), top_data);
    caffe_memset(sizeof(uint) * count, 0, mask);

    const int num_instances = bottom->shape()[0];
    const int num_neurons = bottom->count(1);
	const int k = compute_k(num_neurons);

    for (int n = 0; n < num_instances; n++) {
	    // set up vectors of indices and values for the slice
        std::vector<Dtype> vals;
        vals.assign(bottom_data, bottom_data + num_neurons);
        std::vector<size_t> idxs(vals.size());
        for (size_t i = 0; i < idxs.size(); i++) {
            idxs[i] = i;
        }

        // sort indices descending according to values
        kth_element_idxs_(vals, idxs, k, 0);

        // copy over the top-k values for slice + set mask
        for (size_t i = 0; i < k; i++) {
            top_data[idxs[i]] =  bottom_data[idxs[i]];
            mask[idxs[i]] = static_cast<uint>(1);
        }
		
		// advance all pointers
        mask += num_neurons;
        bottom_data += num_neurons;
        top_data += num_neurons;
    }
}

template <typename Dtype>
void TopKLayer<Dtype>::Forward_cpu_lifetime(const Blob<Dtype>* bottom,
                                     Blob<Dtype>* top) { 
	CHECK_GE(bottom->num_axes(), 2) << "Must have dimensionality of at least 2";
	DCHECK(top->shape() == bottom->shape()) << "Top and bottom must have same shape";

    const Dtype* bottom_data = bottom->cpu_data();
    Dtype* top_data = top->mutable_cpu_data();
    uint* mask = mask_.mutable_cpu_data();
	const int count = bottom->count();

    caffe_set(count, Dtype(0), top_data);
    caffe_memset(sizeof(uint) * count, 0, mask);

    const int num_instances = bottom->shape()[0];
    const int num_neurons = bottom->count(1);
	const int k = compute_k(num_instances);

    for (int j = 0; j < num_neurons; j++) {
	    // set up vectors of indices and values for the slice
        std::vector<Dtype> vals(num_instances);
        std::vector<size_t> idxs(num_instances);
        for (size_t i = 0; i < num_instances; i++) {
			int idx = j + i * num_neurons;
            idxs[i] = idx;
			vals[i] = bottom_data[idx];
        }

        // sort indices descending according to values
        kth_element_idxs_(vals, idxs, k, 0);

        // copy over the top-k values for slice + set mask
        for (size_t i = 0; i < k; i++) {
            top_data[idxs[i]] =  bottom_data[idxs[i]];
            mask[idxs[i]] = static_cast<uint>(1);
        }
    }
}

template <typename Dtype>
void TopKLayer<Dtype>::Forward_cpu_conv_spatial(const Blob<Dtype>* bottom,
                                     Blob<Dtype>* top) {
	CHECK_EQ(bottom->num_axes(), 4) << "Must have dimensionality 4";
	DCHECK(top->shape() == bottom->shape()) << "Top and bottom must have same shape";

    const Dtype* bottom_data = bottom->cpu_data();
    Dtype* top_data = top->mutable_cpu_data();
    uint* mask = mask_.mutable_cpu_data();
	const int count = bottom->count();

    caffe_set(count, Dtype(0), top_data);
    caffe_memset(sizeof(uint) * count, 0, mask);

    const int num_instances = bottom->shape()[0];
    const int num_channels = bottom->shape()[1];
    const int height = bottom->shape()[2];
    const int width = bottom->shape()[3];
	const int area = height * width;
	const int k = compute_k(area);

    for (int n = 0; n < num_instances; n++) {
		for (int c = 0; c < num_channels; c++) {
			std::vector<Dtype> vals(area);
			std::vector<size_t> idxs(area);
			for (int h = 0; h < height; h++) {
			  	for (int w = 0; w < width; w++) {
					int idx = h * width + w;
					idxs[idx] = idx;
					vals[idx] = bottom_data[idx];
				}
			}
			// sort indices descending according to values
			kth_element_idxs_(vals, idxs, k, 0);

			// copy over the top-k values for slice + set mask
			for (size_t i = 0; i < k; i++) {
				top_data[idxs[i]] =  bottom_data[idxs[i]];
				mask[idxs[i]] = static_cast<uint>(1);
			}

			// advance all pointers
			mask += area;
			bottom_data += area;
			top_data += area;
		}
    }
}

template <typename Dtype>
void TopKLayer<Dtype>::Forward_cpu_conv_cross_channel(const Blob<Dtype>* bottom,
                                     Blob<Dtype>* top) { 
	CHECK_EQ(bottom->num_axes(), 4) << "Must have dimensionality 4";
	DCHECK(top->shape() == bottom->shape()) << "Top and bottom must have same shape";

    const Dtype* bottom_data = bottom->cpu_data();
    Dtype* top_data = top->mutable_cpu_data();
    uint* mask = mask_.mutable_cpu_data();
	const int count = bottom->count();

    caffe_set(count, Dtype(0), top_data);
    caffe_memset(sizeof(uint) * count, 0, mask);

    const int num_instances = bottom->shape()[0];
    const int num_channels = bottom->shape()[1];
    const int height = bottom->shape()[2];
    const int width = bottom->shape()[3];
    const int num_neurons = bottom->count(1);
	const int k = compute_k(num_channels);

    for (int n = 0; n < num_instances; n++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				std::vector<Dtype> vals(num_channels);
				std::vector<size_t> idxs(num_channels);
				for (int c = 0; c < num_channels; c++) {
					int idx = ((c * height) + h) * width + w;
					idxs[c] = idx;
					vals[c] = bottom_data[idx];
				}
				// sort indices descending according to values
				kth_element_idxs_(vals, idxs, k, 0);

				// copy over the top-k values for slice + set mask
				for (size_t i = 0; i < k; i++) {
					top_data[idxs[i]] =  bottom_data[idxs[i]];
					mask[idxs[i]] = static_cast<uint>(1);
				}
			}
		}

		// advance all pointer
		mask += num_neurons;
		bottom_data += num_neurons;
		top_data += num_neurons;
	}
}

template <typename Dtype>
void TopKLayer<Dtype>::Forward_cpu_conv_lifetime_channel(const Blob<Dtype>* bottom,
                                     Blob<Dtype>* top) {
	CHECK_EQ(bottom->num_axes(), 4) << "Must have dimensionality 4";
	DCHECK(top->shape() == bottom->shape()) << "Top and bottom must have same shape";

    const Dtype* bottom_data = bottom->cpu_data();
    Dtype* top_data = top->mutable_cpu_data();
    uint* mask = mask_.mutable_cpu_data();
	const int count = bottom->count();

    caffe_set(count, Dtype(0), top_data);
    caffe_memset(sizeof(uint) * count, 0, mask);

    const int num_instances = bottom->shape()[0];
    const int num_channels = bottom->shape()[1];
    const int height = bottom->shape()[2];
    const int width = bottom->shape()[3];
	const int bins = num_instances * height * width;
	const int k = compute_k(bins);

	for (int c = 0; c < num_channels; c++) {
		std::vector<Dtype> vals(bins);
		std::vector<size_t> idxs(bins);
		int i = 0;
		for (int n = 0; n < num_instances; n++) {
			for (int h = 0; h < height; h++) {
			  	for (int w = 0; w < width; w++) {
					int idx = ((n * num_channels + c) * height + h) * width + w;
					idxs[i] = idx;
					vals[i] = bottom_data[idx];
					i++;
				}
			}
		}
		// sort indices descending according to values
		kth_element_idxs_(vals, idxs, k, 0);

		// copy over the top-k values for slice + set mask
		for (size_t i = 0; i < k; i++) {
			top_data[idxs[i]] =  bottom_data[idxs[i]];
			mask[idxs[i]] = static_cast<uint>(1);
		}
    }
}

  template <typename Dtype>
  void TopKLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const unsigned int* mask = mask_.cpu_data();
        const int count = bottom[0]->count();
        for (int i = 0; i < count; ++i) {
            bottom_diff[i] = top_diff[i] * mask[i];
          }
      }
  }

INSTANTIATE_CLASS(TopKLayer);
REGISTER_LAYER_CLASS(TopK);

}  // namespace caffe
