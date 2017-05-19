
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/graph_layers.hpp"
#include "caffe/util/math_functions.hpp"

#define LABEL_00 0
#define LABEL_01 1
#define LABEL_10 2
#define LABEL_11 3
#define LABEL_IGNORE 0

namespace caffe {


template <typename Dtype>
void Graph4CPairLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->channels(), 1) <<
      "Graph4CPair layer input must have 4 channels";

  vector<int> shape;
  shape.push_back(bottom[0]->num());
  shape.push_back(2);
  shape.push_back(bottom[0]->height());
  shape.push_back(bottom[0]->width());

  top[0]->Reshape(shape);
}

template <typename Dtype>
void Graph4CPairLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int spatial_size = height * width;
  
  const Dtype* gt = bottom[0]->cpu_data();
  Dtype* labels = top[0]->mutable_cpu_data();
  for (int n = 0; n < num; n++) {
    int gt_num_offset = n * spatial_size;
    int labels_num_offset = 2 * n * spatial_size;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
  	    int my_spatial_offset = h * height + w;
		Dtype my_val = gt[gt_num_offset + my_spatial_offset];

  	    if (h + 1 < height) {
  	      // down neb
  	      int neb_spatial_offset = (h + 1) * height + w;
  	      Dtype neb_val = gt[gt_num_offset + neb_spatial_offset];
		  Dtype label;
		  if (my_val) {
		    if (neb_val) {
			  label = LABEL_11;
			} else {
			  label = LABEL_10;
			}
		  } else {
		    if (neb_val) {
			  label = LABEL_01;
			} else {
			  label = LABEL_00;
			}
		  }
		  labels[labels_num_offset + my_spatial_offset] = label;
  	    } else {
		  labels[labels_num_offset + my_spatial_offset] = LABEL_IGNORE;
		}
  	    if (w + 1 < width) {
  	      // down neb
  	      int neb_spatial_offset = h * height + w + 1;
  	      Dtype neb_val = gt[gt_num_offset + neb_spatial_offset];
		  Dtype label;
		  if (my_val) {
		    if (neb_val) {
			  label = LABEL_11;
			} else {
			  label = LABEL_10;
			}
		  } else {
		    if (neb_val) {
			  label = LABEL_01;
			} else {
			  label = LABEL_00;
			}
		  }
		  labels[labels_num_offset + spatial_size + my_spatial_offset] = label;
  	    } else {
		  labels[labels_num_offset + spatial_size + my_spatial_offset] = LABEL_IGNORE;
		}
  	  }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(Graph4CPairLayer);
#endif

INSTANTIATE_CLASS(Graph4CPairLayer);
REGISTER_LAYER_CLASS(Graph4CPair);

}  // namespace caffe

#undef LABEL_00
#undef LABEL_10
#undef LABEL_01
#undef LABEL_11
#undef LABEL_IGNORE

