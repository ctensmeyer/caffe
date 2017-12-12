#ifndef CAFFE_IOULOSS_LAYER_HPP_
#define CAFFE_IOULOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/ring.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/multi/geometries/multi_polygon.hpp>

namespace caffe {

template <typename Dtype>
class IntersectionOverUnionLossLayer : public LossLayer<Dtype> {
 public:
  explicit IntersectionOverUnionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), work_buffer_(new Blob<double>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "IntersectionOverUnionLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

 protected:

  typedef boost::geometry::model::d2::point_xy<double> xy;
  typedef boost::geometry::model::polygon<xy,true,false> polygon;
  typedef boost::geometry::model::ring<xy,true,false> ring;
  typedef boost::geometry::model::linestring<xy> linestring;
  typedef boost::geometry::model::multi_polygon<polygon> multi_polygon;

  typedef struct {
    xy p1, p2, p3, p4;
    int i1, i2, i3, i4;
    bool is_original, is_pred;
  } point_meta;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void slope_intercept(xy p1, xy p2, double& m, double& i);
  virtual void d_slope_intercept(xy p1, xy p2, double& dm_dx, 
		double& dm_dy, double& di_dx, double& di_dy);
  virtual void d_inter_d_p1(xy p1, xy p2, xy q1, xy q2, double& dx_dx,
        double& dx_dy, double& dy_dx, double& dy_dy);
  virtual double signed_area(std::vector<xy>& ring);
  virtual void d_area_d_pred(polygon& pred, polygon& gt, multi_polygon& polygons,
  							 double* diffs);
  virtual void find_intersection_points(polygon& pred, polygon& gt, 
  							 std::vector<std::pair<xy,point_meta> >& out);
  virtual void find_point_meta(xy pt, polygon& pred, polygon& gt, 
                             std::vector<std::pair<xy,point_meta> >& intersections,
							 point_meta& out);
  virtual void compute_diff(int idx, std::vector<xy>& ring,
      polygon& pred, polygon& gt, double* diffs, 
	  std::vector<std::pair<xy,point_meta> >& intersections, double sign);

  shared_ptr<Blob<double> > work_buffer_;
};


}  // namespace caffe

#endif 
