#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/intersection_over_union_loss_layer.hpp"

#include <boost/foreach.hpp>


namespace caffe {

template <typename Dtype>
void IntersectionOverUnionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void IntersectionOverUnionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "Must have the same batch size";

  CHECK_GE(bottom[0]->shape(1), 3) << "Predictions must have at least 3 points";
  CHECK_GE(bottom[1]->shape(1), 3) << "GT must have at least 3 points";

  CHECK_EQ(bottom[0]->shape(2), 2) << "Only handles 2D points";
  CHECK_EQ(bottom[1]->shape(2), 2) << "Only handles 2D points";

  vector<int> shape;
  shape.push_back(1);
  shape.push_back(bottom[0]->shape(1));
  shape.push_back(2);

  work_buffer_->Reshape(shape);

  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}

// assumes that pred and gt are simple polygons (no inner rings)
template <typename Dtype>
void IntersectionOverUnionLossLayer<Dtype>::find_intersection_points(polygon& pred, polygon& gt, 
  							 std::vector<std::pair<xy,point_meta> >& out) {
  std::vector<xy> pred_ring = pred.outer(); 
  std::vector<xy> gt_ring = gt.outer(); 
  linestring pred_line_seg;
  linestring gt_line_seg;

  for (int i = 0; i < pred_ring.size(); i++) {
    xy pred_pt1 = pred_ring[i];
    xy pred_pt2 = pred_ring[(i+1) % pred_ring.size()];

    pred_line_seg.clear();
    boost::geometry::append(pred_line_seg, pred_pt1);
    boost::geometry::append(pred_line_seg, pred_pt2);
    for (int j = 0; j < gt_ring.size(); j++) {
      xy gt_pt1 = gt_ring[j];
      xy gt_pt2 = gt_ring[(j+1) % gt_ring.size()];

      gt_line_seg.clear();
      boost::geometry::append(gt_line_seg, gt_pt1);
      boost::geometry::append(gt_line_seg, gt_pt2);

	  bool b = boost::geometry::intersects(pred_line_seg, gt_line_seg);
	  //LOG(INFO) << "Line " << boost::geometry::wkt(pred_line_seg) << " intersects " <<
	  //				boost::geometry::wkt(gt_line_seg) << " : " << b;

	  if (b) {
	    std::vector<xy> intersection_pt;
		point_meta pm;

		boost::geometry::intersection(pred_line_seg, gt_line_seg, intersection_pt);
		CHECK_EQ(1, intersection_pt.size()) << "Should have exactly 1 intersection pt";
		xy pt = intersection_pt[0];

		pm.p1 = pred_pt1;
		pm.i1 = i;
		pm.p2 = pred_pt2;
		pm.i2 = (i+1) % pred_ring.size();

		pm.p3 = gt_pt1;
		pm.i3 = j;
		pm.p4 = gt_pt2;
		pm.i4 = (j+1) % gt_ring.size();

		pm.is_original = false;
		pm.is_pred = false;
		
		out.push_back(std::pair<xy,point_meta>(pt, pm));
	  }
    }
  }
}

template <typename Dtype>
void IntersectionOverUnionLossLayer<Dtype>::find_point_meta(xy pt, polygon& pred, 
  							 polygon& gt, std::vector<std::pair<xy,point_meta> >& intersections,
							 point_meta& out) {
  std::vector<xy> pred_ring = pred.outer(); 
  for (int i = 0; i < pred_ring.size(); i++) {
    xy pred_pt = pred_ring[i];
	if (boost::geometry::equals(pt, pred_pt)) {
	  out.p1 = pred_pt;
	  out.i1 = i;
	  out.is_original = true;
	  out.is_pred = true;
	  return;
	}
  }

  for (int i = 0; i < intersections.size(); i++) {
    std::pair<xy,point_meta> pair = intersections[i];
	if (boost::geometry::equals(pt, pair.first)) {
	  out = pair.second;
	  return;
	}
  }

  std::vector<xy> gt_ring = gt.outer(); 
  for (int j = 0; j < gt_ring.size(); j++) {
    xy gt_pt = gt_ring[j];
	if (boost::geometry::equals(pt, gt_pt)) {
	  out.p1 = gt_pt;
	  out.i1 = j;
	  out.is_original = true;
	  out.is_pred = false;
	  return;
	}
  }

  LOG(FATAL) << "Did not find the point meta";
}

// computes the area before the absolute value (sign needed for derivative of abs)
template <typename Dtype>
double IntersectionOverUnionLossLayer<Dtype>::signed_area(std::vector<xy>& ring) {
  double a = 0;
  for (int i = 0; i < ring.size(); i++) {
    xy pt1 = ring[i];
    xy pt2 = ring[(i+1) % ring.size()];
	a += pt1.x() * pt2.y() - pt1.y() * pt2.x();
  }
  return 0.5 * a;
}

// computes darea_ring/d_ring[idx].  Uses point meta and the chain rule to
// modifify the diffs for all applicable points in pred
template <typename Dtype>
void IntersectionOverUnionLossLayer<Dtype>::compute_diff(int idx, std::vector<xy>& ring,
      polygon& pred, polygon& gt, double* diffs, 
	  std::vector<std::pair<xy,point_meta> >& intersections, double sign) {

  xy pt = ring[idx];
  point_meta meta;
  find_point_meta(pt, pred, gt, intersections, meta); 

  // compute da_dx, da_dy
  int next_idx = (idx + 1) % ring.size();
  int prev_idx = (idx - 1 + ring.size()) % ring.size();
  xy prev_pt = ring[prev_idx];
  xy next_pt = ring[next_idx];
  double da_dx = 0.5 * sign * (next_pt.y() - prev_pt.y());
  double da_dy = 0.5 * sign * (prev_pt.x() - next_pt.x());

  if (meta.is_original) {
    if (meta.is_pred) {
	  // pt is a vertex in the pred polygon.
	  std::vector<xy> pred_ring = pred.outer();

	  int pred_pt_idx = meta.i1;
	  int diff_x_idx = 2 * pred_pt_idx;
	  int diff_y_idx = 2 * pred_pt_idx + 1;

	  diffs[diff_x_idx] += da_dx;
	  diffs[diff_y_idx] += da_dy;
	} else {
	  // pt is a vertex in the gt polygon.  No change to diff
	  return;
	}
  } else {
    // pt is the intersection of two line segments.  
	//   one from pred, one from gt
	std::vector<xy> pred_ring = pred.outer();
	std::vector<xy> gt_ring = gt.outer();

	int pred_pt_idx1 = meta.i1;
	int diff_x_idx1 = 2 * pred_pt_idx1;
	int diff_y_idx1 = 2 * pred_pt_idx1 + 1;
	xy pred_pt1 = pred_ring[pred_pt_idx1];

	int pred_pt_idx2 = meta.i2;
	int diff_x_idx2 = 2 * pred_pt_idx2;
	int diff_y_idx2 = 2 * pred_pt_idx2 + 1;
	xy pred_pt2 = pred_ring[pred_pt_idx2];

	xy gt_pt1 = meta.p3;
	xy gt_pt2 = meta.p4;

	double dx_dx, dx_dy, dy_dx, dy_dy;
	d_inter_d_p1(pred_pt1, pred_pt2, gt_pt1, gt_pt2, 
	             dx_dx, dx_dy, dy_dx, dy_dy);
	diffs[diff_x_idx1] += da_dx * dx_dx + da_dy * dy_dx;
	diffs[diff_y_idx1] += da_dx * dx_dy + da_dy * dy_dy;

	d_inter_d_p1(pred_pt2, pred_pt1, gt_pt1, gt_pt2,
	             dx_dx, dx_dy, dy_dx, dy_dy);
	diffs[diff_x_idx2] += da_dx * dx_dx + da_dy * dy_dx;
	diffs[diff_y_idx2] += da_dx * dx_dy + da_dy * dy_dy;
  }
  // distribute based on point meta
}

template <typename Dtype>
void IntersectionOverUnionLossLayer<Dtype>::slope_intercept(xy p1, xy p2, double& m, double& i) {
  double d_y = p1.y() - p2.y();
  double d_x = p1.x() - p2.x();
  if (d_x == 0) {
    m = std::nan("");
	i = p1.x();  // x-intercept
  } else {
    m = d_y / d_x;  
	i = p1.y() - m * p1.x(); // y-intercept
  }
}

// derivatives wrt p1
// assumes line is not vertical
template <typename Dtype>
void IntersectionOverUnionLossLayer<Dtype>::d_slope_intercept(xy p1, xy p2, double& dm_dx, 
		double& dm_dy, double& di_dx, double& di_dy) {
  double m, i;
  slope_intercept(p1, p2, m, i);

  dm_dx = m / (p2.x() - p1.x());
  dm_dy = -1. / (p2.x() - p1.x());

  di_dx = -1 * p2.x() * dm_dx;
  di_dy = -1 * p2.x() * dm_dy;
}

// dx_dy means the derivative of the x coordinate of the intersection point
//   wrt p1.y()
template <typename Dtype>
void IntersectionOverUnionLossLayer<Dtype>::d_inter_d_p1(xy p1, xy p2, xy q1, xy q2, 
			double& dx_dx, double& dx_dy, double& dy_dx, double& dy_dy) {
  std::vector<xy> intersection_pt;
  linestring p_line_seg;
  boost::geometry::append(p_line_seg, p1);
  boost::geometry::append(p_line_seg, p2);

  linestring q_line_seg;
  boost::geometry::append(q_line_seg, q1);
  boost::geometry::append(q_line_seg, q2);

  boost::geometry::intersection(p_line_seg, q_line_seg, intersection_pt);

  // u.x = (iq - ip) / (mp - mq)
  // u.y = mp * u.x + ip = mq * u.x + iq
  xy u = intersection_pt[0];
  
  double mp, mq, ip, iq;
  slope_intercept(p1, p2, mp, ip);
  slope_intercept(q1, q2, mq, iq);

  // handle vertical line case
  if (std::isnan(mp) || std::isnan(mq)) {
    if (mp == 0 || mq == 0) {
	  if (mp == 0) {
	    // p line is horizontal
		dx_dx = 0;
		dx_dy = 0;
		dy_dx = 0;
		dy_dy = 1;  
		// dy_dy actually should depend on how close
		// p1 and p2 are to the intersection point
		// but this is an edge case, so this is sufficient
		// for gradient based training where this case is
		// really rare
	  } else {
	    // p line is vertical
		dx_dx = 1;
		dx_dy = 0;
		dy_dx = 0;
		dy_dy = 0;
	  }
	} else {
	  // swap x-y coordinates so vertical is now horizontal
	  xy p1r, p2r, q1r, q2r;
	  p1r.x(p1.y()); p1r.y(p1.x());
	  p2r.x(p2.y()); p2r.y(p2.x());
	  q1r.x(q1.y()); q1r.y(q1.x());
	  q2r.x(q2.y()); q2r.y(q2.x());

      // also reverse the x-y components of the output variables
	  // now we can recurse in the new coordinate system that doesn't
	  // have a vertical line
	  d_inter_d_p1(p1r, p2r, q1r, q2r, dy_dy, dy_dx, dx_dy, dx_dx);
	}
  } else {
    // neither line is vertical

    double dmp_dx, dmp_dy, dip_dx, dip_dy;
    d_slope_intercept(p1, p2, dmp_dx, dmp_dy, dip_dx, dip_dy);

    double dx_dmp = -1 * u.x() / (mp - mq);
    double dx_dip = -1 / (mp - mq);
    double dy_dmp = mq * dx_dmp;
    double dy_dip = mq * dx_dip;

    // chain rule
    dx_dx = dx_dmp * dmp_dx + dx_dip * dip_dx;
    dx_dy = dx_dmp * dmp_dy + dx_dip * dip_dy;
    dy_dx = dy_dmp * dmp_dx + dy_dip * dip_dx;
    dy_dy = dy_dmp * dmp_dy + dy_dip * dip_dy;
  }
}

template <typename Dtype>
void IntersectionOverUnionLossLayer<Dtype>::d_area_d_pred(polygon& pred, 
	polygon& gt, multi_polygon& polygons, double* diffs) {
  
  std::vector<std::pair<xy,point_meta> > intersections;
  find_intersection_points(pred, gt, intersections);

  LOG(INFO) << "intersection size: " << intersections.size();
  for (int i = 0; i < intersections.size(); i++) {
    std::pair<xy,point_meta> pair = intersections[i];
	xy p = pair.first;
	point_meta m = pair.second;
	LOG(INFO) << p.x() << "," << p.y() << " " << m.i1 << " " << m.i2 << " " << m.i3 << " " << m.i4;
  }

  double _signed_area;
  double sign;

  BOOST_FOREACH(polygon const& p, polygons) {
    std::vector<xy> outer_ring = p.outer();
	_signed_area = signed_area(outer_ring);
    sign = std::copysign(1., _signed_area); 

	for (int i = 0; i < outer_ring.size(); i++) {
	  compute_diff(i, outer_ring, pred, gt, diffs, intersections, sign);
	}
	const std::vector<ring> inner_rings = p.inners();
	LOG(INFO) << "Inner Rings: " << inner_rings.size();
	for (int j = 0; j < inner_rings.size(); j++) {
	  std::vector<xy> inner_ring = inner_rings[j];
	  _signed_area = signed_area(inner_ring);
	  // inner rings contribute negative area
      sign = -1 * std::copysign(1., _signed_area); 

	  if (boost::geometry::equals(inner_ring[0], inner_ring[inner_ring.size() - 1])) {
        // if ring is closed, make it open
	    inner_ring.pop_back(); 
	  }
	  LOG(INFO) << "Inner Ring size: " << inner_ring.size();
	  for (int i = 0; i < inner_ring.size(); i++) {
		LOG(INFO) << boost::geometry::wkt(inner_ring[i]);
	    compute_diff(i, inner_ring, pred, gt, diffs, intersections, sign);
	  }
	}
  }
}

template <typename Dtype>
void IntersectionOverUnionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const int num = bottom[0]->num();
  const int n_points_pred = bottom[0]->shape(1);
  const int n_points_gt = bottom[1]->shape(1);

  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();

  double total_iou = 0;
  for (int n = 0; n < num; n++) {
    const int offset = 2 * n * n_points_pred;
    polygon pred;
	for (int i = 0; i < n_points_pred; i++) {
      boost::geometry::append(pred, xy(pred_data[offset + 2*i], pred_data[offset + 2*i+1]));
	}

    polygon gt;
	for (int j = 0; j < n_points_gt; j++) {
      boost::geometry::append(gt, xy(gt_data[offset + 2*j], gt_data[offset + 2*j+1]));
	}

	multi_polygon output;
	boost::geometry::intersection(pred, gt, output);
	double intersection_area = boost::geometry::area(output);

	output.clear();
    boost::geometry::union_(pred, gt, output);
	double union_area = boost::geometry::area(output);

	double iou = intersection_area / union_area;
	//double iou = intersection_area;
	total_iou += iou;
  }
  double avg_iou = total_iou / num;
  top[0]->mutable_cpu_data()[0] = (1 - avg_iou);  // need lower to be better
}

template <typename Dtype>
void IntersectionOverUnionLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << " IntersectionOverUnionLossLayer cannot backpropagate to GT input.";
  }
  if (!propagate_down[0]) {
    return;
  }
  const int num = bottom[0]->num();
  const int n_points_pred = bottom[0]->shape(1);
  const int n_points_gt = bottom[1]->shape(1);

  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  Dtype* pred_diff = bottom[0]->mutable_cpu_diff();

  double* intersection_diffs = work_buffer_->mutable_cpu_data();
  double* union_diffs = work_buffer_->mutable_cpu_diff();

  const double d_loss_d_top = top[0]->cpu_diff()[0];

  for (int n = 0; n < num; n++) {
    const int offset = 2 * n * n_points_pred;

    polygon pred;
	for (int i = 0; i < n_points_pred; i++) {
      boost::geometry::append(pred, xy(pred_data[offset+2*i], pred_data[offset+2*i+1]));
	}

    polygon gt;
	for (int j = 0; j < n_points_gt; j++) {
      boost::geometry::append(gt, xy(gt_data[offset+2*j], gt_data[offset+2*j+1]));
	}

	multi_polygon intersection, union_;

	boost::geometry::intersection(pred, gt, intersection);
    boost::geometry::union_(pred, gt, union_);

	const double intersection_area = boost::geometry::area(intersection);
	const double union_area = boost::geometry::area(union_);

    caffe_set(2*n_points_pred, 0., intersection_diffs);
    caffe_set(2*n_points_pred, 0., union_diffs);
    d_area_d_pred(pred, gt, intersection, intersection_diffs);
    d_area_d_pred(pred, gt, union_, union_diffs);

	for (int i = 0; i < 2*n_points_pred; i++) {
      const double d_intersection = intersection_diffs[i];
      const double d_union = union_diffs[i];
	  pred_diff[offset + i] = -1 * d_loss_d_top * (union_area * d_intersection - intersection_area * d_union) /
	  							(union_area * union_area) / num;
	  //pred_diff[offset + i] = -1 * d_loss_d_top * d_intersection;
	}
  }
}

INSTANTIATE_CLASS(IntersectionOverUnionLossLayer);
REGISTER_LAYER_CLASS(IntersectionOverUnionLoss);

}  // namespace caffe
