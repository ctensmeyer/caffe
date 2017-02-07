
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/graph_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "GridCut/GridGraph_2D_4C.h"

// channel layout of input blob
#define SOURCE_CHANNEL 0
#define TERM_CHANNEL 1
#define UD_CHANNEL 2
#define LR_CHANNEL 3

namespace caffe {


template <typename Dtype>
void Graph4CCutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->channels(), 4) <<
      "Graph4CCut layer input must have 4 channels";

  vector<int> shape;
  shape.push_back(bottom[0]->num());
  shape.push_back(1);
  shape.push_back(bottom[0]->height());
  shape.push_back(bottom[0]->width());

  top[0]->Reshape(shape);

  vector<int> shape2;
  shape2.push_back(bottom[0]->num());
  shape2.push_back(1);

  top[1]->Reshape(shape2);
}

template <typename Dtype>
void Graph4CCutLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  typedef GridGraph_2D_4C<Dtype,Dtype,Dtype> Grid;
  
  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int spatial_size = height * width;
  
  const Dtype* edge_weights = bottom[0]->cpu_data();
  Dtype* segmentation = top[0]->mutable_cpu_data();
  Dtype* cuts = top[1]->mutable_cpu_data();
  for (int n = 0; n < num; n++) {
    int num_offset = 4 * num * spatial_size;
    Grid* grid = new Grid(height, width);
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
  	    int spatial_offset = h * height + w;
  
        // source/term weights
  	    Dtype source_edge = edge_weights[num_offset + SOURCE_CHANNEL * spatial_size + spatial_offset];
  	    Dtype term_edge = edge_weights[num_offset + TERM_CHANNEL * spatial_size + spatial_offset];
  	    grid->set_terminal_cap(grid->node_id(h,w), source_edge, term_edge);
  
        // pair weights.  Need to set each direction
  	    if (h > 0) {
  	      // up neb
  	      spatial_offset = (h - 1) * height + w;
  	      Dtype neb_edge = edge_weights[num_offset + UD_CHANNEL * spatial_size + spatial_offset];
  	      grid->set_neighbor_cap(grid->node_id(h,w), -1, 0, neb_edge);
  	    }
  	    if (h + 1 < height) {
  	      // down neb
  	      spatial_offset = h * height + w;
  	      Dtype neb_edge = edge_weights[num_offset + UD_CHANNEL * spatial_size + spatial_offset];
  	      grid->set_neighbor_cap(grid->node_id(h,w), +1, 0, neb_edge);
  	    }
  	    if (w > 0) {
  	      // left neb
  	      spatial_offset = h * height + w - 1;
  	      Dtype neb_edge = edge_weights[num_offset + LR_CHANNEL * spatial_size + spatial_offset];
  	      grid->set_neighbor_cap(grid->node_id(h,w), 0, -1, neb_edge);
  	    }
  	    if (w + 1 < width) {
  	      // right neb
  	      spatial_offset = h * height + w + 1;
  	      Dtype neb_edge = edge_weights[num_offset + LR_CHANNEL * spatial_size + spatial_offset];
  	      grid->set_neighbor_cap(grid->node_id(h,w), 0, +1, neb_edge);
  	    }
  	  }
    }
    grid->compute_maxflow();
  
    int top_num_offset = num * spatial_size;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
  	    int spatial_offset = h * height + w;
  	    segmentation[top_num_offset + spatial_offset] = (Dtype) grid->get_segment(grid->node_id(h,w));
  	  }
    }
    cuts[n] = grid->get_flow();
  }
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(Graph4CCutLayer);
#endif

INSTANTIATE_CLASS(Graph4CCutLayer);
REGISTER_LAYER_CLASS(Graph4CCut);

}  // namespace caffe

#undef SOURCE_CHANNEL
#undef TERM_CHANNEL
#undef UD_CHANNEL
#undef LR_CHANNEL

