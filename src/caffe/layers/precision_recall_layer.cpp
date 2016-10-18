#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PRLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  thresh_ = this->layer_param_.precision_recall_param().thresh();

  has_ignore_label_ =
    this->layer_param_.precision_recall_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.precision_recall_param().ignore_label();
  }
}

template <typename Dtype>
void PRLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << "The two bottom blobs must have the same size to compute precision/recall";

  vector<int> top_shape(0);  // Precision and Recall are scalars; 0 axes.
  top[0]->Reshape(top_shape); // Precision
  top[1]->Reshape(top_shape);  // Recall
  top[2]->Reshape(top_shape);  // Accuracy
  top[3]->Reshape(top_shape);  // True Negative Rate
  top[4]->Reshape(top_shape);  // F-measure
}

template <typename Dtype>
void PRLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype true_positives = 0, false_positives = 0, true_negatives = 0, false_negatives = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  for (int i = 0; i < bottom[0]->count(); ++i) {
    const int label_value =
        static_cast<int>(bottom_label[i]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      continue;
    }
	const Dtype actual_value = bottom_data[i];
	if (actual_value >= thresh_) {
	  // predicted positive
	  if (label_value) {
	    // actually positive
		true_positives++;
	  } else {
	    // actually negative
	    false_positives++;
	  }
	} else {
	  // predicted negative
	  if (label_value == 0) {
	    // actually negative
		true_negatives++;
	  } else {
	    // actually positive
	    false_negatives++;
	  }
	}
  }

  Dtype precision = 0;
  if (true_positives + false_positives != 0) {
    precision = true_positives / (true_positives + false_positives);  // precision
  }
  Dtype recall = 0;
  if (true_positives + false_negatives != 0) {
    recall = true_positives / (true_positives + false_negatives);  // recall
  }
  top[0]->mutable_cpu_data()[0] = precision;
  top[1]->mutable_cpu_data()[0] = recall;
  top[2]->mutable_cpu_data()[0] = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives);  // accuracy
  top[3]->mutable_cpu_data()[0] = true_negatives / (false_positives + true_negatives);  // true negative rate
  if (precision + recall != 0) {
    top[4]->mutable_cpu_data()[0] = (2 * precision * recall) / (precision + recall);
  } else {
    top[4]->mutable_cpu_data()[0] = 0;
  }

  // PR layer should not be used as a loss function.
}

INSTANTIATE_CLASS(PRLayer);
REGISTER_LAYER_CLASS(PR);

}  // namespace caffe
