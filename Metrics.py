import numpy as np
import json
import pickle
import csv
import random


def load_idx2u():
    with open('/kaggle/working/GCN/data/r_MOOC10000/idx2u.pickle', 'rb') as f:
        return pickle.load(f)


def load_u2idx():
    with open('/kaggle/working/GCN/data/r_MOOC10000/u2idx.pickle', 'rb') as f:
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

        scores = {'hits@' + str(k): [] for k in k_list}
        scores.update({'map@' + str(k): [] for k in k_list})
        for p_, y_ in zip(y_prob, y_true):
            if y_ != self.PAD:
                scores_len += 1.0
                p_sort = p_.argsort()
                for k in k_list:
                    topk = p_sort[-k:][::-1]
                    scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
                    scores['map@' + str(k)].extend([self.apk([y_], topk, k)])

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len

    def get_courses_by_video(self, video_name, course_video_mapping):
        """根据视频名称获取其所属的课程"""
        courses = []
        for course, videos in course_video_mapping.items():
            if video_name in videos:
                courses.append(course)
        return courses

    def compute_metric_pro(self, y_prob, y_true, y_prev, w_c, d_t, w_t, d_1, d_2, d_3, k_list=[5, 10, 20]):
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
        u2idx = load_u2idx()
        courses = load_course()
        prev_course_list = []

        scores = {f'hits@{k}': [] for k in k_list}
        scores.update({f'map@{k}': [] for k in k_list})

        # 提取课程对应视频的映射关系
        course_video_mapping = self.build_course_video_mapping(courses)

        for p_, y_, y_p, wc, dt, wt, d1, d2, d3 in zip(y_prob, y_true, y_prev, w_c, d_t, w_t, d_1, d_2, d_3):
            if y_ == self.PAD:
                prev_course_list = []
                continue

            scores_len += 1
            initial_topk = self.get_top_k_predictions(p_, k=40)
            topk_course_list = []
            topk_diversity_video = []
            prev_diversity_video = []

            prev_video_name = idx2u[y_p]
            prev_courses = self.get_courses_by_video(prev_video_name, course_video_mapping)
            if prev_courses[0] not in prev_course_list:
                prev_course_list.insert(0, prev_courses[0])

            next_video_id = None


            # 计算预测视频的分数
            scores_pro, f_next_video = self.score_predictions(initial_topk, y_p, idx2u, course_video_mapping, courses, prev_courses)


            # 根据得分重新排序topk
            sorted_topk = self.reorder_top_predictions(initial_topk, scores_pro)
            # sorted_top20 = sorted_topk[:20]
            for v in sorted_topk:
                topk_v_name = idx2u[v]
                topk_course = self.get_courses_by_video(topk_v_name, course_video_mapping)
                topk_course_list.append(topk_course[0])


            topk_diversity_video = self.random_videos_from_courses(topk_course_list, course_video_mapping, 2, seed=58)
            prev_diversity_video = self.random_videos_from_courses(prev_course_list, course_video_mapping, 2, seed=58)

            sorted_topk = sorted_topk.insert(20, topk_diversity_video, prev_diversity_video)



            if f_next_video:
                # 通过前一个视频找到相邻的下一个视频
                # prev_video_name = idx2u[y_p]
                next_video_id = self.find_next_video(prev_video_name, prev_courses, u2idx, courses)
            # 如果找到 next_video_id，则将其插入到首位
            if next_video_id is not None:
                sorted_topk.insert(0, next_video_id)

            # prev_video = self.find_prev_video(prev_video_name, prev_courses, u2idx, courses)
            # if prev_video is not None:
            #     for v in prev_video:
            #         if v not in sorted_topk:
            #             sorted_topk.insert(0, next_video_id)

            # 更新结果
            for k in k_list:
                topk = sorted_topk[:k]
                scores[f'hits@{k}'].append(1.0 if y_ in topk else 0.0)
                scores[f'map@{k}'].append(self.apk([y_], topk, k))

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len

    def find_next_video(self, prev_video_name, prev_courses, u2idx, courses):
        """在课程中找到相邻的下一个视频"""
        # prev_courses = self.get_courses_by_video(prev_video_name, course_video_mapping)
        for course_id in prev_courses:
            for course in courses:
                if course['id'] == course_id:
                    video_order = course.get('video_order', [])
                    try:
                        y_index = video_order.index(prev_video_name)
                        # 如果下一个视频存在，返回它的ID
                        if y_index + 1 < len(video_order):
                            next_video_name = video_order[y_index + 1]
                            if next_video_name in u2idx:
                                return u2idx[next_video_name]
                    except ValueError:
                        continue
        return None

    def build_course_video_mapping(self, courses):
        """构建课程与视频的映射关系，减少重复查找"""
        mapping = {}
        for course in courses:
            course_id = course['id']
            video_order = course.get('video_order', [])
            mapping[course_id] = video_order
        return mapping

    def get_top_k_predictions(self, p_, k=20):
        """获取排序后的前K个预测视频"""
        return p_.argsort()[::-1][:k]

    def score_predictions(self, topk, y_p, idx2u, course_video_mapping, courses, prev_courses):
        """根据与历史视频的关联性给每个预测视频评分"""
        # scores_pro = {video_id: 0 for video_id in topk}
        scores_pro = {video_id: (20-i) if i < 20 else 0 for i, video_id in enumerate(topk)}
        prev_video_name = idx2u[y_p]

        for video_id in topk:
            predicted_video_name = idx2u[video_id]
            predicted_courses = self.get_courses_by_video(predicted_video_name, course_video_mapping)
            # prev_courses = self.get_courses_by_video(prev_video_name, course_video_mapping)

            # 计算距离评分
            score, f_next_video = self.calculate_distance_score(predicted_courses, prev_courses, courses,
                                                                  prev_video_name, predicted_video_name)

            scores_pro[video_id] += score

        return scores_pro, f_next_video

    def calculate_distance_score(self, predicted_courses, prev_courses, courses, prev_video_name, predicted_video_name):
        """计算课程内的相对视频位置的距离评分"""
        score = 0
        f_next_video = True  # 新增一个标志，判断是否需要单独找相邻视频

        for pred_course in predicted_courses:
            for prev_course in prev_courses:
                if pred_course == prev_course:
                    course_info = next((c for c in courses if c['id'] == pred_course), None)
                    if course_info:
                        video_order = course_info.get('video_order', [])
                        try:
                            y_index = video_order.index(prev_video_name)
                            pred_index = video_order.index(predicted_video_name)
                            distance = pred_index - y_index

                            if distance == 1:
                                score += 10  # 确保相邻视频加足够高的分数
                                f_next_video = False  # 标记为不需要再找下一个视频
                            elif distance == -1:
                                score += 10
                            elif abs(distance) == 2:
                                score += 10
                            elif abs(distance) == 3:
                                score += 10
                            elif abs(distance) == 4:
                                score += 10
                        except ValueError:
                            continue

        # # 如果确实找到了相邻视频，可以在这里设置更高的优先级
        # if f_next_video:
        #     score += 10  # 再次强化相邻视频的分数

        return score, f_next_video


    def reorder_top_predictions(self, topk, scores_pro):
        """根据得分重新排序topk预测视频，优先将下一个视频插入到第一位"""
        scored_videos = sorted(((v, scores_pro[v]) for v in topk if scores_pro[v] > 0), key=lambda x: x[1],
                               reverse=True)
        unscored_videos = [v for v in topk if scores_pro[v] == 0]

        sorted_videos = [v for v, _ in scored_videos] + unscored_videos

        return sorted_videos

    def random_videos_from_courses(self, course_list, course_video_mapping, num_videos=3, seed=None):
        """
        从 topk_course_list 中的每个课程在 course_video_mapping 中随机抽取 num_videos 个视频（不重复），可以设置种子。
        """
        selected_videos = []

        if seed is not None:
            random.seed(seed)  # 设置随机种子

        # for course_id in course_list:
        for index, course_id in enumerate(course_list):
            if index == 3:
                break
            if course_id in course_video_mapping:
                videos = course_video_mapping[course_id]
                # 从视频列表中随机抽取 num_videos 个不重复的视频
                selected_videos.extend(random.sample(videos, min(len(videos), num_videos)))  # 确保不会超出列表长度

        return selected_videos


    def find_prev_video(self, prev_video_name, prev_courses, u2idx, courses):
        """在课程中找到相邻的下一个视频"""
        # prev_courses = self.get_courses_by_video(prev_video_name, course_video_mapping)
        prev_video_list = []
        prev_video = []
        for course_id in prev_courses:
            for course in courses:
                if course['id'] == course_id:
                    video_order = course.get('video_order', [])
                    try:
                        y_index = video_order.index(prev_video_name)
                        for i in range(4, 0, -1):
                            if (y_index - i) >= 0:
                                prev_video_list.append(video_order[y_index - i])

                            if (y_index + i) < len(video_order):
                                prev_video_list.append(video_order[y_index + i])

                        for v in prev_video_list:
                            if v in u2idx:
                                prev_video.append(u2idx[v])
                        return prev_video

                    except ValueError:
                        continue
        return None





