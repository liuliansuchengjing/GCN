
import numpy as np
import json
import pickle

# 从idx2u.pickle中加载索引到名称的映射
def load_idx2u():
    with open('/kaggle/working/GCN/data/r_MOOC10000/idx2u.pickle', 'rb') as f:
        return pickle.load(f)


# 从course - video.json中加载课程 - 视频关系
def load_course_video():  
    try:  
        with open('path_to_your_json_file.json', 'r', encoding='utf-8') as f:  
            return json.load(f)  
    except FileNotFoundError:  
        print("指定的文件未找到，请检查文件路径。")  
    except json.JSONDecodeError:  
        print("文件内容不是有效的 JSON 格式，请检查文件内容。")  
    except Exception as e:  
        print(f"发生了一个错误：{e}")


# 从course.json中加载课程数据
def load_course():
    with open('/kaggle/input/riginmooccube/MOOCCube/entities/course.json', 'r', encoding='utf-8') as f:
        return json.load(f)

class Video:
    def __init__(self, index, course):
        self.index = index
        self.course = course

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
		# 加载数据
		idx2u = load_idx2u()
		course_video = load_course_video()
		courses = load_course()


		scores = {'hits@'+str(k):[] for k in k_list}
		scores.update({'map@'+str(k):[] for k in k_list})
		for p_, y_, y_p, c_p in zip(y_prob, y_true, y_prev, course_prev):
			if y_ != self.PAD:
				scores_len += 1.0
				p_sort_desc = p_.argsort()[::-1]
				for k in k_list:
					top100 = p_sort_desc[:k ]
					scores = {video_id: 0 for video_id in top100}
					for video_id in top100:
						# 获取预测视频的名称
						predicted_video_name = idx2u[video_id]
						# 获取预测视频所属的课程
						predicted_video_courses = []
						for course, videos in course_video.items():
							if video_id in videos:
								predicted_video_courses.append(course)

						for predicted_course in predicted_video_courses:
							if predicted_course == c_p:
								y_video_name = idx2u[y_p]
								# 在每个课程的video_order中查找y_video_name和predicted_video_name的位置
								for course in courses:
									if course['id'] == predicted_course:
										video_order = course['video_order']
										try:
											y_index = video_order.index(y_video_name)
											predicted_index = video_order.index(predicted_video_name)
											distance = abs(y_index - predicted_index)
											if distance == 1:
												scores[video_id] += 100
											elif distance == 2:
												scores[video_id] += 50
											elif distance == 3:
												scores[video_id] += 20
										except ValueError:
											pass

					# 根据得分对top100重新排序
					# 得分的视频及其分数
					scored_videos = [(video_id, score) for video_id, score in scores.items() if score > 0]
					# 按照分数从高到低排序得分的视频
					scored_videos.sort(key=lambda pair: pair[1], reverse=True)
					scored_video_ids = [video_id for video_id, _ in scored_videos]

					# 未得分的视频
					unscored_video_ids = [video_id for video_id in top100 if video_id not in scores or scores[video_id] == 0]

					# 合并结果
					sorted_top100 = scored_video_ids + unscored_video_ids


					topk = sorted_top100[:k]
					scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
					scores['map@'+str(k)].extend([self.apk([y_], topk, k)])


		scores = {k: np.mean(v) for k, v in scores.items()}
		return scores, scores_len


