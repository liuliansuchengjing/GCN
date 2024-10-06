
import numpy as np
import json
import pickle
import csv


def load_idx2u():
	with open('/kaggle/working/GCN/data/r_MOOC10000/idx2u.pickle', 'rb') as f:
		return pickle.load(f)


def load_course_video():
	data = {}
	with open('/kaggle/input/riginmooccube/MOOCCube/relations/course-video.json', 'r', encoding='utf-8') as file:
		reader = csv.reader(file, delimiter='\t')
		for row in reader:
			if len(row) == 2:
				course_id, video_id = row
				if course_id not in data:
					data[course_id] = []
				data[course_id].append(video_id)
	return data


def get_topk(score_seq, k):
	p_sort_desc = score_seq.argsort()[::-1]

	topk = p_sort_desc[:k]
	# print("1.topk:", topk)
	return topk


def course_order_score(prevent_video, hyper_topk):
	# 加载数据
	idx2u = load_idx2u()
	course_video = load_course_video()
	courses = load_course()

	scores_pro = {video_id: 0 for video_id in hyper_topk}
	for video_id in hyper_topk:
		# 获取预测视频的名称
		y_video_name = idx2u[prevent_video]
		predicted_video_name = idx2u[video_id]
		predicted_video_courses = []
		prev_course = []
		for course, videos in course_video.items():
			if predicted_video_name in videos:
				predicted_video_courses.append(course)
			if y_video_name in videos:
				prev_course.append(course)

		for predicted_course in predicted_video_courses:
			for c_p in prev_course:
				if predicted_course == c_p:
					for course in courses:
						if course['id'] == predicted_course:
							video_order = course['video_order']
							try:
								for index, video_name in enumerate(video_order):
									if video_name == y_video_name:
										y_index = index
									if video_name == predicted_video_name:
										predicted_index = index
								distance = abs(y_index - predicted_index)
								# print("distance:", distance)

								if distance == 1:
									scores_pro[video_id] += 10
								elif distance == 2:
									scores_pro[video_id] += 9
								elif distance == 3:
									scores_pro[video_id] += 8
								elif distance == 4:
									scores_pro[video_id] += 5
							except ValueError:
								pass

	return scores_pro


def topk_reorder(scores_pro, topk):
	# 根据得分对top20重新排序
	# 得分的视频及其分数
	scored_videos = [(video_id, score_pro) for video_id, score_pro in scores_pro.items() if score_pro > 0]
	# 按照分数从高到低排序得分的视频
	scored_videos.sort(key=lambda pair: pair[1], reverse=True)
	scored_video_ids = [video_id for video_id, _ in scored_videos]
	# 未得分的视频
	unscored_video_ids = [video_id for video_id in topk if video_id not in scores_pro or scores_pro[video_id] == 0]
	# 合并结果
	sorted_topk = scored_video_ids + unscored_video_ids

	return sorted_topk



def load_course():
	courses = []
	with open('/kaggle/input/riginmooccube/MOOCCube/entities/course.json', 'r', encoding='utf-8') as f:
		data = f.read()
		start = 0
		end = 0
		while True:
			start = data.find('{', end)
			if start == - 1:
				break
			end = data.find('}', start) + 1
			json_obj = data[start:end]
			try:
				course = json.loads(json_obj)
				courses.append(course)
			except json.decoder.JSONDecodeError as e:
				print(f"解析错误: {e}")
	return courses


class Metrics(object):

	def __init__(self):
		super().__init__()
		self.PAD = 0

	def apk(self, actual, predicted, k=10):
		"""
		Computes the average precision at k.
		This function computes the average prescision at k between two lists of
		items.
		Parameters
		----------
		actual : list
				 A list of elements that are to be predicted (order doesn't matter)
		predicted : list
					A list of predicted elements (order does matter)
		k : int, optional
			The maximum number of predicted elements
		Returns
		-------
		score : double
				The average precision at k over the input lists
		"""
		score = 0.0
		num_hits = 0.0

		for i, p in enumerate(predicted):
			if p in actual and p not in predicted[:i]:
				num_hits += 1.0
				score += num_hits / (i + 1.0)

		# if not actual:
		# 	return 0.0
		return score / min(len(actual), k)


	def compute_metric(self, y_prob, y_true, k_list=[10, 50, 100]):
		'''
			y_true: (#samples, )
			y_pred: (#samples, #users)
		'''
		scores_len = 0
		y_prob = np.array(y_prob)
		y_true = np.array(y_true)

		scores = {'hits@'+str(k):[] for k in k_list}
		scores.update({'map@'+str(k):[] for k in k_list})
		for p_, y_ in zip(y_prob, y_true):
			if y_ != self.PAD:
				scores_len += 1.0
				p_sort = p_.argsort()
				for k in k_list:
					topk = p_sort[-k:][::-1]
					scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
					scores['map@'+str(k)].extend([self.apk([y_], topk, k)])

		scores = {k: np.mean(v) for k, v in scores.items()}
		return scores, scores_len

	def compute_metric_pro(self, y_prob, y_true, y_prev, course_prev, k_list=[5, 10, 20]):
		'''
			y_true: (#samples, )
			y_pred: (#samples, #users)
		'''
		scores_len = 0
		y_prob = np.array(y_prob)
		y_true = np.array(y_true)
		y_prev = np.array(y_prev)
		y_list = []

		scores = {'hits@'+str(k):[] for k in k_list}
		scores.update({'map@'+str(k):[] for k in k_list})
		for p_, y_, y_p in zip(y_prob, y_true, y_prev):

			if y_ != self.PAD:
				scores_len += 1.0
				y_list.append(y_)
				top20 = get_topk(p_, 20)
				scores_pro = course_order_score(y_p, top20)
				sorted_top20 = topk_reorder(scores_pro, top20)

				for k in k_list:
					topk = sorted_top20[:k]
					# if k == 20:
					# 	if y_ not in topk:
					# 		# 打印 topk 和 y_
					# 		print("topk:", topk)
					# 		print("y_list:", y_list)

					
					# print("topk:", topk)
					scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
					scores['map@'+str(k)].extend([self.apk([y_], topk, k)])

			else:
				y_list.clear()

		scores = {k: np.mean(v) for k, v in scores.items()}
		return scores, scores_len
