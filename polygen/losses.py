import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # Hide TF deprecation messages
import matplotlib.pyplot as plt


import modules
import data_utils
from quality_utils import *


# class MeshQuality(object):
# 	def __init__(self, mesh):
# 		self.vertices = mesh['vertices']
# 		self.faces = mesh['faces']


# 	def quad_total_ratio(self):
# 		total = len(self.faces)
# 		quads = len([face for face in self.faces if len(face) == 4])
# 		return float(quads) / total


# 	def max_aspect_ratio(self):
# 		def aspect_ratio(face_verts):
# 			if len(face_verts) > 4:
# 				return 1000.
# 			ds = [distance(face_verts[i], face_verts[i + 1]) for i in range(len(face_verts) - 1)]
# 			ds.append(distance(face_verts[-1], face_verts[0]))
# 			return tf.reduce_max(ds) / tf.reduce_min(ds)
# 		_max_aspect_ratio = -1.
# 		for _face in self.faces:
# 			verts = self.vertices[_face]
# 			_max_aspect_ratio = tf.reduce_max([_max_aspect_ratio, aspect_ratio(verts)])
# 		return _max_aspect_ratio


# 	def max_skewness(self):
# 		def skewness(face_verts):
# 			if len(face_verts) > 4:
# 				return 1.
# 			if len(face_verts) == 4:
# 				theta_e = 90 # check angle v radian
# 			elif len(face_verts) == 3:
# 				theta_e = 60
# 			else:
# 				raise ValueError
# 			_angles = angles(face_verts)
# 			theta_max = tf.reduce_max(_angles)
# 			theta_min = tf.reduce_min(_angles)
# 			return tf.reduce_max([(theta_max - theta_e) / (180 - theta_e), (theta_min - theta_e) / theta_e])
# 		_max_skewness = -1.
# 		for _face in self.faces:
# 			verts = self.vertices[_face]
# 			_max_skewness = tf.reduce_max([_max_skewness, skewness(verts)])
# 		return _max_skewness


# 	def min_face_angle(self):
# 		_min_face_angle = 180.
# 		for _face in self.faces:
# 			verts = self.vertices[_face]
# 			_min_face_angle = min(_min_face_angle, min(angles(verts)))
# 		return _min_face_angle


# 	def size(self):
# 		pass


# def qtr(faces):
# 	quad = 0.0
# 	total = 0.0
# 	for face in faces:
# 		total += 1.0
# 		if len(face) == 4:
# 			quad += 1.0
# 	return quad / total


def qtr(flat_faces):
	#print('QTR')
	running = 0
	quads = 0.
	total = 0.
	for ff in flat_faces[:-1]:
		#print(running, ff, quads, total)
		if ff > 0 and ff < 2: # for some reason, ff == 1 does not work
			if running == 4:
				quads += 1.
			if running > 2:
				total += 1.
			running = 0
		running += 1
	if running > 2:
		total += 1.
	if total < 1:
		return 0.
	return 1. - quads / total



def quad_total_ratio(f):
	# total = tf.cast(tf.shape(f)[0], tf.float32)
	# quad_locs = tf.where_v2(tf.size(f) == 4)[0, :]
	# n_quads = tf.cast(tf.size(quad_locs), tf.float32)
	# return n_quads / total
	return tf.py_function(qtr, [f], tf.float32)


