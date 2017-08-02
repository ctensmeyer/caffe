
import random
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from math import pi

DEBUG = False


def slope_intercept(p1, p2):
	'''
	p1 = (x,y), p2 = (x,y)
	return (slope, intercept)
	If the line is non-vertical, slope is a real value and intercept
		is the y-intercept
	If the line is vertical, slope will be None and intercept is the
		x-intercept
	'''
	d_y = p1[1] - p2[1]
	d_x = p1[0] - p2[0]
	slope = d_y / d_x if d_x != 0 else None
	if slope is None:
		intercept = p1[0]
	else:
		intercept = p1[1] - slope * p1[0] if slope is not None else None
		if DEBUG:
			intercept2 = p2[1] - slope * p2[0] if slope is not None else None
			if intercept and intercept2:
				#print intercept, intercept2, p1, p2, slope
				assert np.isclose(intercept, intercept2, atol=0.01)

	return slope, intercept


def line_intersect(m1, i1, m2, i2):
	'''
	Returns the point of intersection of two lines described
		by slope and intercept
	Returns None if lines are parallel or the same line
	'''
	if m1 == m2:
		return None
	if m1 is not None and m2 is not None:
		x_inter = (i2 - i1) / (m1 - m2)
		y_inter = m1 * x_inter + i1
		if DEBUG:
			y_inter2 = m2 * x_inter + i2
			#print (m1, i1), (m2, i2), y_inter, y_inter2
			assert np.isclose(y_inter, y_inter2)
		return (x_inter, y_inter)

	# exactly one line is vertical
	if m2 is None:
		# swap so that m1 is vertical
		m1, i1, m2, i2 = m2, i2, m1, i1

	assert m1 is None
	x_inter = i1
	y_inter = m2 * x_inter + i2
	return (x_inter, y_inter)


def between(c, a, b, rtol=1e-07, atol=1e-10):
	'''
	return True is c is between a and b.  Accepts both a < b and b > a
	'''
	eps = atol + rtol * abs(c)
	return (((a - eps) <= c and c <= (b + eps)) or 
	        ((b - eps) <= c and c <= (a + eps)))


def d_slope_intercept(p1, p2):
	'''
	Assumes p1[0] != p2[0] (line is not vertical)
	'''
	m, i = slope_intercept(p1, p2)
	# m = (y2 - y1) / (x2 - x1)
	dm_dp1x = m / (p2[0] - p1[0])
	dm_dp1y = -1 / (p2[0] - p1[0])
	dm_dp2x = -1 * dm_dp1x  # by symmetry
	dm_dp2y = -1 * dm_dp1y  # by symmetry

	# i = y1 - m * x1 = y2 - m * x2
	# we get to pick which equation, so for simplicity,
	di_dp1x = -1 * p2[0] * dm_dp1x  # i = y2 - m * x2
	di_dp1y = -1 * p2[0] * dm_dp1y  # i = y2 - m * x2
	di_dp2x = -1 * p1[0] * dm_dp2x  # i = y1 - m * x1
	di_dp2y = -1 * p1[0] * dm_dp2y  # i = y1 - m * x1

	if DEBUG:
		# check against computing di_dp1x using i = y1 - m * x1
		_di_dp1x = -1 * (dm_dp1x * p1[0] + m)
		_di_dp1y = 1 - (p1[0] * dm_dp1y)
		assert np.isclose(_di_dp1x, di_dp1x)
		assert np.isclose(_di_dp1y, di_dp1y)

	return ( (dm_dp1x, di_dp1x), (dm_dp1y, di_dp1y) ), ( (dm_dp2x, di_dp2x), (dm_dp2y, di_dp2y) ) 


def d_line_seg_intersection_vertical(p1, p2, q1, q2):
	mp, ip = slope_intercept(p1, p2)
	mq, iq = slope_intercept(q1, q2)

	if mq is None:
		# flip arguments so that p1 <-> p2 is vertical
		du = d_line_seg_intersection_vertical(q1, q2, p1, p2)

		# flip outputs back to match original inputs
		return (du[2], du[3], du[0], du[1])

	# at this point, mp is None, meaning p1 <-> p2 is vertical

	if mq != 0:
		# swap coords so that the line is no longer vertical
		_p1, _p2 = (p1[1], p1[0]), (p2[1], p2[0])
		_q1, _q2 = (q1[1], q1[0]), (q2[1], q2[0])
		
		# compute derivatives in the swaped coord space
		(((dux_dp1x, duy_dp1x), (dux_dp1y, duy_dp1y)),
		 ((dux_dp2x, duy_dp2x), (dux_dp2y, duy_dp2y)),
		 ((dux_dq1x, duy_dq1x), (dux_dq1y, duy_dq1y)),
		 ((dux_dq2x, duy_dq2x), (dux_dq2y, duy_dq2y))) = d_line_seg_intersection(_p1, _p2, _q1, _q2)

		# swap coords back
		return (((dux_dp1y, duy_dp1y), (dux_dp1x, duy_dp1x)),
				((dux_dp2y, duy_dp2y), (dux_dp2x, duy_dp2x)),
				((dux_dq1y, duy_dq1y), (dux_dq1x, duy_dq1x)),
				((dux_dq2y, duy_dq2y), (dux_dq2x, duy_dq2x)))

	else:
		# we have a vertical line (p1 <-> p2) intersecting a horizontal line (q1 <-> q2)
		return (((1., 0.), (0., 0.)),
				((1., 0.), (0., 0.)),
				((0., 0.), (0., 1.)),
				((0., 0.), (0., 1.)))


def d_line_seg_intersection(p1, p2, q1, q2):
	'''
	Assumes the two line segments actually intersect
	'''
	mp, ip = slope_intercept(p1, p2)
	mq, iq = slope_intercept(q1, q2)

	u = line_intersect(mp, ip, mq, iq)
	if u is None:
		# lines don't intersect, returns 0
		return (((0., 0.), (0., 0.)),
				((0., 0.), (0., 0.)),
				((0., 0.), (0., 0.)),
				((0., 0.), (0., 0.)))

	if mp is None or mq is None:
		return d_line_seg_intersection_vertical(p1, p2, q1, q2)

	(((dmp_dp1x, dip_dp1x), (dmp_dp1y, dip_dp1y) ), 
	( (dmp_dp2x, dip_dp2x), (dmp_dp2y, dip_dp2y) )) = d_slope_intercept(p1, p2)

	(((dmq_dq1x, diq_dq1x), (dmq_dq1y, diq_dq1y) ), 
	( (dmq_dq2x, diq_dq2x), (dmq_dq2y, diq_dq2y) )) = d_slope_intercept(q1, q2)

	# ux = (iq - ip) / (mp - mq)
	dux_dmp = -1 * u[0] / (mp - mq)
	dux_dip = -1 / (mp - mq)
	dux_dmq = -1 * dux_dmp
	dux_diq = -1 * dux_dip

	# uy = mp * ux + ip = mq * ux + iq
	# we get to pick which equation, so for simplicity,
	duy_dmp = mq * dux_dmp  # uy = mq * ux + iq
	duy_dip = mq * dux_dip  # uy = mq * ux + iq
	duy_dmq = mp * dux_dmq  # uy = mp * ux + ip
	duy_diq = mp * dux_diq  # uy = mp * ux + ip

	# u[0] derivatives
	dux_dp1x = dux_dmp * dmp_dp1x + dux_dip * dip_dp1x
	dux_dp1y = dux_dmp * dmp_dp1y + dux_dip * dip_dp1y
	dux_dp2x = dux_dmp * dmp_dp2x + dux_dip * dip_dp2x
	dux_dp2y = dux_dmp * dmp_dp2y + dux_dip * dip_dp2y

	dux_dq1x = dux_dmq * dmq_dq1x + dux_diq * diq_dq1x
	dux_dq1y = dux_dmq * dmq_dq1y + dux_diq * diq_dq1y
	dux_dq2x = dux_dmq * dmq_dq2x + dux_diq * diq_dq2x
	dux_dq2y = dux_dmq * dmq_dq2y + dux_diq * diq_dq2y

	# u[1] derivatives
	duy_dp1x = duy_dmp * dmp_dp1x + duy_dip * dip_dp1x
	duy_dp1y = duy_dmp * dmp_dp1y + duy_dip * dip_dp1y
	duy_dp2x = duy_dmp * dmp_dp2x + duy_dip * dip_dp2x
	duy_dp2y = duy_dmp * dmp_dp2y + duy_dip * dip_dp2y

	duy_dq1x = duy_dmq * dmq_dq1x + duy_diq * diq_dq1x
	duy_dq1y = duy_dmq * dmq_dq1y + duy_diq * diq_dq1y
	duy_dq2x = duy_dmq * dmq_dq2x + duy_diq * diq_dq2x
	duy_dq2y = duy_dmq * dmq_dq2y + duy_diq * diq_dq2y

	return (((dux_dp1x, duy_dp1x), (dux_dp1y, duy_dp1y)),
		    ((dux_dp2x, duy_dp2x), (dux_dp2y, duy_dp2y)),
		    ((dux_dq1x, duy_dq1x), (dux_dq1y, duy_dq1y)),
		    ((dux_dq2x, duy_dq2x), (dux_dq2y, duy_dq2y)))


def line_seg_intersect(p1, p2, q1, q2):
	'''
	Given two line segments described by two pairs of endpoints,
		return the point of intersection.  If they do not intersect,
		return None.
	'''
	p_slope, p_intercept = slope_intercept(p1, p2)
	#print p_slope, p_intercept
	q_slope, q_intercept = slope_intercept(q1, q2)
	#print q_slope, q_intercept
	p_intersect = line_intersect(p_slope, p_intercept, q_slope, q_intercept)
	#print p_intersect
	if p_intersect is None:
		return p_intersect
	if (between(p_intersect[0], p1[0], p2[0]) and
	    between(p_intersect[1], p1[1], p2[1]) and
	    between(p_intersect[0], q1[0], q2[0]) and
	    between(p_intersect[1], q1[1], q2[1])):
		return p_intersect
	return None  


def below_line(p, m, i):
	'''
	Return True if p is on or below the line defined by (m, i).
		If (m, i) is vertical, return True if p is on or to the
		left of the line.
	'''
	if m is not None:
		y_line = m * p[0] + i
		return p[1] <= y_line
	else:
		return p[0] <= i


def same_side(p1, p2, m, i):
	return below_line(p1, m, i) == below_line(p2, m, i)
	

def point_inside_polygon(p, V):
	'''
	Return true if p is inside or on the border of the polygon
		defined by the ordered list of points V.  V is assumed
		to be convex and non-degenerate (no 3 colinear vertices)
	Check each side of the polygon.  If p is inside in the polygon,  
		p must be on the same side of the line defined by the side as 
		all other vertices of the polygon.
	'''
	for i in xrange(len(V)):
		j = (i - 1) % len(V)
		k = (i + 1) % len(V)
		p1 = V[j]
		p2 = V[i]
		p3 = V[k]

		m, inter = slope_intercept(p1, p2) 
		if not same_side(p, p3, m, inter):
			return False
	return True
	

def area(V):
	return abs(_area_helper(V))


def _area_helper(V):
	'''
	V is a list of vertices of a polygon (assumed to be ordered)
	'''
	a = 0
	for i in xrange(len(V)):
		j = (i - 1) % len(V)
		p1 = V[j]
		p2 = V[i]
		a += p1[0] * p2[1] - p1[1] * p2[0] 
	return 0.5 * a


def d_area(V, mult=1.0):
	sign = mult * np.sign(_area_helper(V))

	D = list()
	for i in xrange(len(V)):
		j = (i - 1) % len(V)
		k = (i + 1) % len(V)
		p1 = V[j]
		p2 = V[k]

		d_x = 0.5 * sign * (p2[1] - p1[1])
		d_y = 0.5 * sign * (p1[0] - p2[0])

		D.append( (d_x, d_y) )
	return D
	

def dist(p, q):
	return np.sqrt( (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def find_closest_point_idx(p, qs):
	min_dist = np.inf
	min_idx = -1
	for idx, q in enumerate(qs):
		if q is None:
			continue
		d = dist(p, q)
		if d < min_dist:
			min_dist = d
			min_idx = idx
	return min_idx
	

def next_int(v, direction):
	if isinstance(v, (int, long)):
		return int(v + direction)
	if isinstance(v, float):
		if v.is_integer():
			return int(v + direction)
		if direction > 0:
			return int(np.ceil(v))
		else:
			return int(np.floor(v))


def pt_is_close(p1, p2, atol=1e-10, rtol=1e-8):
	return (np.isclose(p1[0], p2[0], rtol=rtol, atol=atol) and 
	   		np.isclose(p1[1], p2[1], rtol=rtol, atol=atol))


def contains_poly(V1, V2):
	''' return true if V1 contains V2 '''
	for p in V2:
		if not point_inside_polygon(p, V1):
			return False
	return True


def area_subtract_polygons(V1, V2):
	''' Gets the area of the residual of V1 - V2 '''
	if contains_poly(V1, V2):
		return area(V1) - area(V2)
	elif contains_poly(V2, V1):
		return 0.
	else:
		polygons, _ = subtract_polygons(V1, V2)
		total_area = 0
		for polygon in polygons:
			total_area += area(polygon)
		return total_area


def d_area_subtract_polygons(V1, V2):
	''' Gets (d_area of the residual of V1 - V2)/d_V1,d_V2 '''
	if contains_poly(V1, V2):
		#print "Contains"
		d_V1 = d_area(V1)
		d_V2 = d_area(V2, mult=-1.)
		return d_V1, d_V2
	elif contains_poly(V2, V1):
		d_V1 = [[0, 0] for _ in xrange(len(V1))] 
		d_V2 = [[0, 0] for _ in xrange(len(V2))] 
		return d_V1, d_V2
	else:
		return _d_area_subtract_polygons(V1, V2)


def _d_area_subtract_polygons(V1, V2):
	polygons, metas = subtract_polygons(V1, V2)

	d_V1 = [[0, 0] for _ in xrange(len(V1))] 
	d_V2 = [[0, 0] for _ in xrange(len(V2))] 

	for polygon, meta in zip(polygons, metas):
		# get the derivative of area wrt all coords
		da = d_area(polygon)
		for idx in xrange(len(polygon)):
			# iterate over the vertices of the polygon
			meta_info = meta[idx]
			da_dx = da[idx][0]  # derivative of area wrt the x coord of the vertex
			da_dy = da[idx][1]  # derivative of area wrt the y coord of the vertex
			if len(meta_info) == 2:
				# this vertex is exactly a vertex in V1 or V2 
				part_of_V1 = meta_info[1]
				pt_idx = meta_info[0]
				if part_of_V1:
					d_V1[pt_idx][0] += da_dx
					d_V1[pt_idx][1] += da_dy
				else:
					d_V2[pt_idx][0] += da_dx
					d_V2[pt_idx][1] += da_dy
			elif len(meta_info) == 4:
				# this vertex resulted from the intersection of 2 line segments

				# pull from the meter info the vertex indices for the 
				# line segments causing the intersection
				v1_idx_1 = meta_info[0]
				v1_idx_2 = meta_info[1]
				v2_idx_1 = meta_info[2]
				v2_idx_2 = meta_info[3]

				# get the derivative of the intersection coord wrt 
				# all 4 points (16 values total)
				d_inter = d_line_seg_intersection(V1[v1_idx_1], V1[v1_idx_2], V2[v2_idx_1], V2[v2_idx_2])
				d_dv11, d_dv12, d_dv21, d_dv22 = d_inter

				# unpack
				(dx_dv11x, dy_dv11x), (dx_dv11y, dy_dv11y) = d_dv11
				(dx_dv12x, dy_dv12x), (dx_dv12y, dy_dv12y) = d_dv12
				(dx_dv21x, dy_dv21x), (dx_dv21y, dy_dv21y) = d_dv21
				(dx_dv22x, dy_dv22x), (dx_dv22y, dy_dv22y) = d_dv22


				# apply chain rule
				d_V1[v1_idx_1][0] += da_dx * dx_dv11x + da_dy * dy_dv11x
				d_V1[v1_idx_1][1] += da_dx * dx_dv11y + da_dy * dy_dv11y

				d_V1[v1_idx_2][0] += da_dx * dx_dv12x + da_dy * dy_dv12x
				d_V1[v1_idx_2][1] += da_dx * dx_dv12y + da_dy * dy_dv12y

				d_V2[v2_idx_1][0] += da_dx * dx_dv21x + da_dy * dy_dv21x
				d_V2[v2_idx_1][1] += da_dx * dx_dv21y + da_dy * dy_dv21y

				d_V2[v2_idx_2][0] += da_dx * dx_dv22x + da_dy * dy_dv22x
				d_V2[v2_idx_2][1] += da_dx * dx_dv22y + da_dy * dy_dv22y


			else:
				raise Exception('Bad Meta: %r' % meta_info)

	return d_V1, d_V2


def subtract_polygons(V1, V2):
	'''
	returns a list of list of (x,y) tuples representing the polygons 
		created by subtracting V2 from V1.  Assumes V1 and V2 are ordered
		lists of polygons.
	'''
	polygons = list()
	metas = list()
	visited = [False] * len(V1)
	for idx in xrange(len(V1)):
		if point_inside_polygon(V1[idx], V2):
			# this vertex cannot be part of a residual polygon
			visited[idx] = True

	if DEBUG:
		print V1
		print V2
		print visited

	i = 0
	max_iters = 2 * (len(V1) + len(V2)) + 1
	while not all(visited):
		start_pt_idx = visited.index(False)
		start_pt = V1[start_pt_idx]
		cur_pt_idx = start_pt_idx
		cur_pt = start_pt
		cur_pt_is_intersection = False
		if DEBUG:
			print "start_point: ", start_pt

		direction = 1
		polygon = list()
		meta = list()
		first = True
		cur_points = V1
		other_points = V2
		meta.append( (start_pt_idx, True) )
		iters = 0
		while first or cur_pt != start_pt:
			# bookkeeping
			first = False
			polygon.append(cur_pt)
			if isinstance(cur_pt_idx, int) and cur_pt == V1[cur_pt_idx]:
				visited[cur_pt_idx] = True
			if DEBUG:
				print "\ncur_point: ", cur_pt

			# attempt to move to the next vertex on the current quad
			next_pt_idx = next_int(cur_pt_idx, direction) % len(cur_points) 
			potential_next_pt = cur_points[next_pt_idx]
			cur_m, cur_i = slope_intercept(cur_pt, potential_next_pt)
			if DEBUG: 
				print "potential_next_pt: ", potential_next_pt

			# find any intersections
			intersections = map(lambda idx: line_seg_intersect(cur_pt, 
								potential_next_pt, other_points[idx], 
								other_points[(idx+1)%len(other_points)]), xrange(len(other_points)))
			if cur_pt_is_intersection:
				for idx in xrange(len(other_points)):
					# If cur_pt is already an intersection point, remove it
					if intersections[idx] and pt_is_close(intersections[idx], cur_pt):
						intersections[idx] = None
			if DEBUG:
				print "intersections: ", intersections
			if not any(intersections):
				# no intersections, move to next vertex
				if DEBUG: 
					print "no intersections"
				cur_pt_idx = next_pt_idx
				cur_pt = potential_next_pt
				cur_pt_is_intersection = False
				meta.append( (cur_pt_idx, V1 is cur_points) )
			else:
				# move to the closest intersection point
				closest_intersection_idx = find_closest_point_idx(cur_pt, 
														   intersections)

				other_cur_quad_vertex_idx = 0
				other_endpoint_idx = next_int(next_pt_idx, -1 * direction) % len(cur_points)
				#print next_pt_idx, other_endpoint_idx
				while (other_cur_quad_vertex_idx == other_endpoint_idx  or
					   other_cur_quad_vertex_idx == next_pt_idx):
					   other_cur_quad_vertex_idx += 1

				# the intersection point is between 
				# other_points[closest_intersection_idx] and 
				# other_points[(closest_intersection_idx+1)%4]
				cur_pt_idx = closest_intersection_idx + 0.5
				cur_pt = intersections[closest_intersection_idx]
				cur_pt_is_intersection = True
				if DEBUG:
					print "Intersect with:", cur_pt
					print "Between:", other_points[closest_intersection_idx], other_points[(closest_intersection_idx+1)%len(other_points)]
				if V1 is cur_points:
					meta.append( (next_pt_idx, other_endpoint_idx, closest_intersection_idx, 
								 (closest_intersection_idx+1)%len(other_points)) )
				else:
					meta.append( (closest_intersection_idx, (closest_intersection_idx+1)%len(other_points),
								  next_pt_idx, other_endpoint_idx) )

				# the direction needs to be towards the end point of the 
				# intersecting line segment that lies inside the Quad self 
				# or outside the Quad other, depending on which one is
				# cur_points
				#print "Side Check: ", cur_points[other_cur_quad_vertex_idx], other_points[closest_intersection_idx]
				if same_side(cur_points[other_cur_quad_vertex_idx], 
							 other_points[closest_intersection_idx],
							 cur_m, cur_i):
					if DEBUG:
						print "Same Side"
					direction = -1
				else:
					direction = 1

				if V2 is cur_points:
					if DEBUG:
						print "Reversing direction"
					direction *= -1
				if DEBUG:
					print 'cur idx, direction:', cur_pt_idx, direction

				# iterate on the other polygon
				cur_points, other_points = other_points, cur_points
			#if i >= 1:
			#	exit()
			#	break
			i += 1
			iters += 1
			if iters > max_iters:
				print polygons
				raise Exception("Not Terminating (%d steps) with inputs %r,%r" % (iters, V1, V2))
		polygons.append(polygon)
		del meta[-1]
		metas.append(meta)
		#break
	return polygons, metas


def is_poly_ordered(V):
	# examine all pairs of non-adjacent pairs of edges to see if they intersect
	for idx1 in xrange(len(V)):
		p1 = V[idx1]
		p2 = V[(idx1+1) % len(V)]
		end_idx2 = len(V)
		if idx1 == 0:
			# When on the edge 0 -> 1, don't compare
			# with the edge (len(V)-1) -> 0
			end_idx2 -= 1
		for idx2 in xrange(idx1+2, end_idx2):
			q1 = V[idx2]
			q2 = V[(idx2+1) % len(V)]
			inter = line_seg_intersect(p1, p2, q1, q2)
			if inter is not None:
				return False
	return True


def is_poly_convex(V):
	if len(V) == 3:
		return True
	
	v = V[1:]
	for idx in xrange(len(V)):
		if point_inside_polygon(V[idx], v):
			return False
		# replace next vertex with previous one
		# % is to prevent index error on last iteration
		v[idx % len(v)] = V[idx]

	return True


def is_poly_convex_unordered(V):
	if len(V) == 3:
		return True

	V = np.asarray(V)
	hull = scipy.spatial.ConvexHull(V)

	return hull.vertices.shape[0] == V.shape[0]


def angle_between(p1, p2):
	''' returns the angle in [0, 2pi] from p1 to p2 '''
	slope = slope_intercept(p1, p2)[0]
	if slope is None:
		# vertical line
		if p2[1] > p1:
			angle = pi/2
		else:
			angle = 3 * pi/2
	elif slope >= 0:
		if p2[0] > p1[0]:
			# Q1 (p2 is in first quadrant relative to an origin at p1)
			angle = np.arctan(slope)
		else:
			# Q3
			angle = pi + np.arctan(slope)
	else:
		if p2[0] > p1[0]:
			# Q4
			angle = 2 * pi + np.arctan(slope)
		else:
			# Q2
			angle = pi + np.arctan(slope)

	return angle


def get_polygon_order(V):
	# works on convex polygons
	total_x, total_y = 0., 0.
	for idx in xrange(len(V)):
		total_x += V[idx][0]
		total_y += V[idx][1]
	avg_x = total_x / len(V)
	avg_y = total_y / len(V)
	avg_p = (avg_x, avg_y)
	angles = map(lambda v: angle_between(avg_p, v), V)
	return np.argsort(angles)


class Quad():
	
	# ps is [(x,y)] * 4
	def __init__(self, ps):
		if DEBUG:
			assert len(ps) == 4
		self.points = ps[:]
		self.make_ordered()
		#assert self.is_convex()
	
	def is_ordered(self):
		inter_01_23 = line_seg_intersect(self.points[0], self.points[1],
										 self.points[2], self.points[3])
		inter_12_30 = line_seg_intersect(self.points[1], self.points[2],
										 self.points[3], self.points[0])
		return inter_01_23 is None and inter_12_30 is None

	def make_ordered(self):
		if not self.is_ordered():
			tmp = self.points[2]
			self.points[2] = self.points[1]
			self.points[1] = tmp
		if not self.is_ordered():
			self.points = [self.points[0], self.points[2],
						   self.points[3], self.points[1]]
		if not self.is_ordered():
			raise Exception("3 orderings didn't work")

	def is_convex(self):
		'''
		We are convex if no vertex is inside the the triangle formed
			by the other 3 vertices
		'''
		return (not point_inside_polygon(self.points[0], [self.points[1],
									 self.points[2], self.points[3]]) and
			   not point_inside_polygon(self.points[1], [self.points[0],
									 self.points[2], self.points[3]]) and
			   not point_inside_polygon(self.points[2], [self.points[0],
									 self.points[1], self.points[3]]) and
			   not point_inside_polygon(self.points[3], [self.points[0],
									 self.points[1], self.points[2]]))

	def contains_point(self, p):
		return point_inside_polygon(p, self.points)

	def area(self):
		return area(self.points)

	def d_area(self):
		return d_area(self.points)

	def contains_quad(self, other):
		return (self.contains_point(other.points[0]) and
		        self.contains_point(other.points[1]) and
		        self.contains_point(other.points[2]) and
		        self.contains_point(other.points[3]))

	def difference(self, other):
		return subtract_polygons(self.points, other.points)

	def d_area_difference(self, other):
		return _d_area_subtract_polygons(self.points, other.points)

	def area_difference(self, other):
		return area_subtract_polygons(self.points, other.points)

	def step_towards(self, other, lr=0.01, alpha=1, beta=1):
		'''
		alpha is the penalty on self - other
		beta is the penalty on other - self

		'''
		if self.contains_quad(other):
			da = self.d_area()
			for idx in xrange(4):
				# decrease area
				self.points[idx] = (self.points[idx][0] - lr * alpha * da[idx][0], 
									self.points[idx][1] - lr * alpha * da[idx][1])
		elif other.contains_quad(self):
			da = self.d_area()
			for idx in xrange(4):
				# increase area
				self.points[idx] = (self.points[idx][0] + lr * beta * da[idx][0], 
									self.points[idx][1] - lr * beta * da[idx][1])
		else:
			# interesting case
			smo_da_ds, smo_da_do = self.d_area_difference(other)
			oms_da_do, oms_da_ds = other.d_area_difference(self)

			for idx in xrange(4):
				self.points[idx] = (self.points[idx][0] - lr * (alpha * smo_da_ds[idx][0] + beta * oms_da_ds[idx][0]) , 
									self.points[idx][1] - lr * (alpha * smo_da_ds[idx][1] + beta * oms_da_ds[idx][1]))



def plot(points, color='blue'):
	for idx in xrange(len(points)):
		p1 = points[idx]
		p2 = points[(idx+1)%len(points)]
		x, y = [p1[0], p2[0]], [p1[1], p2[1]]
		plt.plot(x,y, marker='o', color=color)


def gradient_descent(q1, q2):
	lr = 0.25
	for step in xrange(300):
		print step
		print q1.points
		print q2.points
		print
		if step % 10 == 0:
			plot(q1.points, color='red')
			plot(q2.points, color='blue')
			plt.show()
			plt.clf()
			lr *= 0.9
		q1.step_towards(q2, lr=lr)


def one():
	#points_1 = [(50., 50.), (400., 50.), (400., 400.), (50., 400.)]
	#points_2 = [(200., 200.), (600., 200.), (600., 600.), (200., 600.)]
	#points_2 = [(200., 0.), (300., 200.), (200., 500.), (0., 200.)]
	points_1, points_2 = [(0.9432091116905212, 0.8659855723381042), (0.01772836595773697, 0.8639823794364929), (0.02373798005282879, 0.08994390815496445), (0.9402043223381042, 0.07892628014087677)],[(0.9299576282501221, 0.8622260093688965), (0.015937695279717445, 0.8531931638717651), (0.01449277251958847, 0.07618695497512817), (0.9402621388435364, 0.09529659152030945)]
	
	print is_poly_ordered(points_1)
	print is_poly_ordered(points_2)
	print is_poly_convex(points_1)
	print is_poly_convex(points_2)
	#print line_seg_intersect(points_2[0], points_2[3], points_1[2], points_1[3])
	#q1 = Quad(points_1)
	#q2 = Quad(points_2)
	#gradient_descent(q1, q2)
	plot(points_1)
	plot(points_2, color='green')
	#plt.show()
	#exit()
	q1_q2, q1_q2_metas = subtract_polygons(points_1, points_2)
	#q2_q1, q2_q1_metas = subtract_polygons(points_2, points_1)
	idx = 0
	colors = ['red', 'orange', 'cyan', 'black']
	for poly, meta in zip(q1_q2, q1_q2_metas):
		if poly:
			print poly
			print meta
			plot(poly, color=colors[idx])
			idx += 1
			print area(poly)
			print
	
	#idx = 0
	#colors = ['yellow', 'brown', 'purple', 'gray']
	#for poly, meta in zip(q2_q1, q2_q1_metas):
	#	if poly:
	#		print poly
	#		print meta
	#		plot(poly, color=colors[idx])
	#		idx += 1
	#		print area(poly)
	#		print
	plt.show()

def two():
	from polygon_area_layer import reorder
	count = 0
	while count < 100000:
		V = list()
		for idx in xrange(4):
			V.append( (random.random(),random.random()) )
		if is_poly_convex_unordered(V):
			count += 1
			V_ordered, _ = reorder(V)
			if not is_poly_ordered(V_ordered):
				print count, V, V_ordered
				break
			
			
			
def three():
	V1 = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]
	V2 = [(0.80000001, 0.80000001), (0.2, 0.89999998), (0.1, 0.2), (0.69999999, 0.1)]
	print d_area(V1)
	print d_area(V2)
	a, b = d_area_subtract_polygons(V1, V2)
	print a
	print b

	a, b = d_area_subtract_polygons(V2, V1)
	print a
	print b

def four():
	p1 = (0.94321436, 0.86789978)
	p2 = (0.040307812, 0.8990944)
	q1 = (0.94320911, 0.86798877)
	q2 = (0.9375, 0.082932696)
	print line_seg_intersect(p1, p2, q1, q2)


if __name__ == "__main__":
	one()
	#two()
	#three()
	#four()





