import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # Hide TF deprecation messages


@tf.function
def subtract(p1, p2):
	return [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]


@tf.function
def midpoint(p1, p2):
	x1, y1, z1 = p1
	x2, y2, z2 = p2
	return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, (z1 + z2) / 2.0]


@tf.function
def distance(p1, p2):
	x1, y1, z1 = p1
	x2, y2, z2 = p2
	d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
	return tf.math.sqrt(d2)


@tf.function
def norm(vec):
	return distance(vec, (0, 0, 0))


@tf.function
def dot_product(vec1, vec2):
	return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


@tf.function
def angle(p1, p2, p3):
	vec1 = subtract(p2, p1)
	vec2 = subtract(p3, p2)
	norm1 = norm(vec1)
	norm2 = norm(vec2)
	dot = dot_product(vec1, vec2)
	return tf.math.arccos(dot / (norm1 * norm2))


@tf.function
def angles(face_verts):
	_angles = [angle(face_verts[i], face_verts[i + 1], face_verts[i + 2]) for i in range(len(face_verts) - 2)]
	_angles.append(angle(face_verts[-2], face_verts[-1], face_verts[0]))
	_angles.append(angle(face_verts[-1], face_verts[0], face_verts[1]))
	return _angles
