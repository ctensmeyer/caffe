#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxFull2ForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* target_data, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	loss[index] = -target_data[index] * log(max(prob_data[index], Dtype(FLT_MIN)));
  }
}

template <typename Dtype>
void SoftmaxFullLoss2Layer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* target_data = bottom[1]->gpu_data();
  const int nthreads = prob_.count();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  SoftmaxFull2ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, target_data, loss_data);
      
  Dtype loss;
  for (int i = 0; i < outer_num_; i++) {
    caffe_gpu_asum(inner_num_, loss_data + i * inner_num_, &loss);
    if (normalize_) {
      loss /= prob_.count();
    } else {
      loss /= outer_num_;
    }
    top[0]->mutable_cpu_data()[i] = loss;
  }

  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxFullLoss2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to target probs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* target_probs = bottom[1]->gpu_data();

	caffe_gpu_sub(prob_.count(), prob_data, target_probs, bottom_diff);

    for (int i = 0; i < outer_num_; i++) {
      const Dtype loss_weight = top[0]->cpu_diff()[i];
	  Dtype mult = 0;
      if (normalize_) {
	    mult = loss_weight / prob_.count();
      } else {
	    mult = loss_weight / outer_num_;
      }
      caffe_gpu_scal(inner_num_, mult, bottom_diff + i * inner_num_);
	}
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxFullLoss2Layer);

}  // namespace caffe
