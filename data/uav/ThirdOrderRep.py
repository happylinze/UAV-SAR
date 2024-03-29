import numpy as np

class ThirdOrderRep:
	def __init__(self):  # C * T * V * M   [3, T, 25, 2]
		self.local_linkage = [
			# head
			# 0: nose  1: left_eye 2: right_eye 3: left_ear 4: right_ear
			(1, 0, 2), (3, 1, 0), (4, 2, 0), (0, 0, 0), (0, 0, 0),
			# up torso with arms
			# 5: left_shoulder 6: right_shoulder 7: left_elbow 8: right_elbow 9: left_wrist 10: right_wrist
			(7, 5, 11), (8, 6, 12), (9, 7, 5), (10, 8, 6), (0, 0, 0), (0, 0, 0),
			# down torso with legs
			# 11: left_hip 12: right_hip 13: left_knee 14: right_knee 15: left_ankle 16: right_ankle
			(13, 11, 12), (14, 12, 11), (15, 13, 11), (16, 14, 12), (0, 0, 0), (0, 0, 0)
		]
		self.pairwise_linkage = [
			#  hands, elbows, knees feet
			[(9, 0, 10), (7, 0, 8), (13, 0, 14), (15, 0, 16)],
			[(9, 1, 10), (7, 1, 8), (13, 1, 14), (15, 1, 16)],
			[(9, 2, 10), (7, 2, 8), (13, 2, 14), (15, 2, 16)],
			[(9, 3, 10), (7, 3, 8), (13, 3, 14), (15, 3, 16)],
			[(9, 4, 10), (7, 4, 8), (13, 4, 14), (15, 4, 16)],
			[(9, 5, 10), (7, 5, 8), (13, 5, 14), (15, 5, 16)],
			[(9, 6, 10), (7, 6, 8), (13, 6, 14), (15, 6, 16)],
			[(9, 7, 10), (0, 0, 0), (13, 7, 14), (15, 7, 16)],
			[(9, 8, 10), (0, 0, 0), (13, 8, 14), (15, 8, 16)],
			[(0, 0, 0), (7, 9, 8), (13, 9, 14), (15, 9, 16)],
			[(0, 0, 0), (7, 10, 8), (13, 10, 14), (15, 10, 16)],
			[(9, 11, 10), (7, 11, 8), (13, 11, 14), (15, 11, 16)],
			[(9, 12, 10), (7, 12, 8), (13, 12, 14), (15, 12, 16)],
			[(9, 13, 10), (7, 13, 8), (0, 0, 0), (15, 13, 16)],
			[(9, 14, 10), (7, 14, 8), (0, 0, 0), (15, 14, 16)],
			[(9, 15, 10), (7, 15, 8), (13, 15, 14), (0, 0, 0)],
			[(9, 16, 10), (7, 16, 8), (13, 16, 14), (0, 0, 0)],
		]
		self.torso_linkage = [
			# left_shoulder -> right_shoulder -> right_hip -> left_hip
			[(5, 0, 6), (6, 0, 12), (12, 0, 11), (11, 0, 5)],
			[(5, 1, 6), (6, 1, 12), (12, 1, 11), (11, 1, 5)],
			[(5, 2, 6), (6, 2, 12), (12, 2, 11), (11, 2, 5)],
			[(5, 3, 6), (6, 3, 12), (12, 3, 11), (11, 3, 5)],
			[(5, 4, 6), (6, 4, 12), (12, 4, 11), (11, 4, 5)],
			[(0, 0, 0), (6, 5, 12), (12, 5, 11), (0, 0, 0)],
			[(0, 0, 0), (0, 0, 0), (12, 6, 11), (11, 6, 5)],
			[(5, 7, 6), (6, 7, 12), (12, 7, 11), (11, 7, 5)],
			[(5, 8, 6), (6, 8, 12), (12, 8, 11), (11, 8, 5)],
			[(5, 9, 6), (6, 9, 12), (12, 9, 11), (11, 9, 5)],
			[(5, 10, 6), (6, 10, 12), (12, 10, 11), (11, 10, 5)],
			[(5, 11, 6), (6, 11, 12), (0, 0, 0), (0, 0, 0)],
			[(5, 12, 6), (0, 0, 0), (0, 0, 0), (11, 12, 5)],
			[(5, 13, 6), (6, 13, 12), (12, 13, 11), (11, 13, 5)],
			[(5, 14, 6), (6, 14, 12), (12, 14, 11), (11, 14, 5)],
			[(5, 15, 6), (6, 15, 12), (12, 15, 11), (11, 15, 5)],
			[(5, 16, 6), (6, 16, 12), (12, 16, 11), (11, 16, 5)]
		]

	def _calculate_angle(self, frame_data, joint_unit):
		# frame_data   3 x 17 x 2
		skeleton_num = frame_data.shape[2]

		angle_rep = np.zeros(skeleton_num)
		if joint_unit == (0, 0, 0):
			return angle_rep

		for m in range(skeleton_num):
			# for outer node1
			x_temp_out = frame_data[0, joint_unit[0], m]
			y_temp_out = frame_data[1, joint_unit[0], m]
			z_temp_out = frame_data[2, joint_unit[0], m]
			outer_node1 = np.array([x_temp_out, y_temp_out, z_temp_out])

			x_temp_out = frame_data[0, joint_unit[1], m]
			y_temp_out = frame_data[1, joint_unit[1], m]
			z_temp_out = frame_data[2, joint_unit[1], m]
			center_node = np.array([x_temp_out, y_temp_out, z_temp_out])

			x_temp_out = frame_data[0, joint_unit[2], m]
			y_temp_out = frame_data[1, joint_unit[2], m]
			z_temp_out = frame_data[2, joint_unit[2], m]
			outer_node2 = np.array([x_temp_out, y_temp_out, z_temp_out])

			bone_vec1 = outer_node1 - center_node
			bone_vec2 = outer_node2 - center_node

			angle_rep[m] = self._included_angle_rep(bone_vec1, bone_vec2)

		return angle_rep

	def getThridOrderRep(self, frames_data):
		# frames_data   3 x frame_num x 17 x 2
		frame_num = frames_data.shape[1]
		skeleton_num = frames_data.shape[3]
		ThridOrderRepfeature = np.zeros((9, frame_num, 17, skeleton_num))

		for frame_index in range(frame_num):
			frame_data = frames_data[:, frame_index, :, :]

			for joint_idx in range(17):
				# local_linkage
				ThridOrderRepfeature[0, frame_index, joint_idx, :] = \
					self._calculate_angle(frame_data, self.local_linkage[joint_idx])
				# pair_wise
				ThridOrderRepfeature[1, frame_index, joint_idx, :] = \
					self._calculate_angle(frame_data, self.pairwise_linkage[joint_idx][0])
				ThridOrderRepfeature[2, frame_index, joint_idx, :] = \
					self._calculate_angle(frame_data, self.pairwise_linkage[joint_idx][1])
				ThridOrderRepfeature[3, frame_index, joint_idx, :] = \
					self._calculate_angle(frame_data, self.pairwise_linkage[joint_idx][2])
				ThridOrderRepfeature[4, frame_index, joint_idx, :] = \
					self._calculate_angle(frame_data, self.pairwise_linkage[joint_idx][3])
				# torso
				ThridOrderRepfeature[5, frame_index, joint_idx, :] = \
					self._calculate_angle(frame_data, self.torso_linkage[joint_idx][0])
				ThridOrderRepfeature[6, frame_index, joint_idx, :] = \
					self._calculate_angle(frame_data, self.torso_linkage[joint_idx][1])
				ThridOrderRepfeature[7, frame_index, joint_idx, :] = \
					self._calculate_angle(frame_data, self.torso_linkage[joint_idx][2])
				ThridOrderRepfeature[8, frame_index, joint_idx, :] = \
					self._calculate_angle(frame_data, self.torso_linkage[joint_idx][3])

		return ThridOrderRepfeature

	def _included_angle_rep(self, vec1, vec2):
		dotProd = np.dot(vec1, vec2)
		normProd = np.sqrt(np.dot(vec1, vec1) * np.dot(vec2, vec2))
		if normProd == 0:
			return 0

		cosined_angle = dotProd / normProd
		# included_ang = np.arccos(cosined_angle)
		rep = 1 - cosined_angle
		return rep

	# def _cross_prod(self, vec1, vec2):
	# 	# b1c2-b2c1,c1a2-a1c2,a1b2-a2b1
	# 	x = vec1[1] * vec2[2] - vec2[1] * vec1[2]
	# 	y = vec1[2] * vec2[0] - vec2[2] * vec1[0]
	# 	z = vec1[0] * vec2[1] - vec2[0] * vec1[1]
	# 	return np.array([x, y, z])
	#
	# def _norm_vec(self, vec1, vec2):
	# 	unnormed_vec = self._cross_prod(vec1, vec2)
	# 	norm = np.sqrt(unnormed_vec.dot(unnormed_vec))
	# 	return unnormed_vec / norm

# test1 = np.array([1, 1, 0])
# test2 = np.array([0, 1, 1])
# print(norm_vec(test1, test2))