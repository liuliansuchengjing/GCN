import numpy as np
import json
import pickle
import csv
import random
from collections import defaultdict
import networkx as nx


class StudentWatchData:
    def __init__(self, video_name, video_course, watch_count, watch_time, total_time):
        self.video_name = video_name
        self.video_course = video_course
        self.watch_count = watch_count
        self.watch_time = watch_time
        self.total_time = total_time

    def calculate_mastery(self):
        """
        计算学生对该视频的掌握度 (观看时长 / 视频总时长)
        :return: 掌握度 (0到1之间的浮点数)
        """
        if self.total_time == 0:
            return 0
        return self.watch_time / self.total_time

    def research_course_order(self, courses):
        course_info = next((c for c in courses if c['id'] == self.video_course), None)
        if course_info:
            video_order = course_info.get('video_order', [])
            y_index = video_order.index(self.video_name)
        return y_index

    def __repr__(self):
        """
        定义类的字符串表示，方便调试和打印
        """
        return f"video_name={self.video_name}, watch_count={self.watch_count}, watch_time={self.watch_time}, total_time={self.total_time}, video_course={self.video_course}"


class ConceptGraph:
    def __init__(self, video_concept_file, parent_son_file):
        self.video_concept_file = video_concept_file
        self.parent_son_file = parent_son_file
        self.video_concept_mapping = self.load_video_concept_mapping()

    def load_video_concept_mapping(self):
        """加载视频-概念映射关系，存储为字典，减少重复 I/O 操作"""
        video_concept_mapping = {}
        with open(self.video_concept_file, 'r', encoding='utf-8') as file:
            for line in file:
                video, concept = line.strip().split('\t')
                if video not in video_concept_mapping:
                    video_concept_mapping[video] = []
                video_concept_mapping[video].append(concept)
        return video_concept_mapping

    def find_focus_concept(self, last_video):
        """通过查找加载好的视频-概念映射，快速找到指定视频的概念"""
        # print("(find)last_video:", last_video)
        consept = self.video_concept_mapping.get(last_video, [])
        # print("(find)consept:", consept)
        return consept

    def draw_knowledge_graph(self):
        # 创建知识图谱，包括视频-概念和父子关系
        knowledge_graph = nx.DiGraph()

        # 添加父子概念关系（无向边）
        with open(self.parent_son_file, 'r', encoding='utf-8') as file:
            for line in file:
                parent, child = line.strip().split('\t')
                if parent not in knowledge_graph:
                    knowledge_graph.add_node(parent, type='concept')
                if child not in knowledge_graph:
                    knowledge_graph.add_node(child, type='concept')
                knowledge_graph.add_edge(parent, child)
                knowledge_graph.add_edge(child, parent)  # 使父子关系无向

        # 添加视频与概念的关系（有向边）
        for video, concepts in self.video_concept_mapping.items():
            if video not in knowledge_graph:
                knowledge_graph.add_node(video, type='video')
            for concept in concepts:
                if concept not in knowledge_graph:
                    knowledge_graph.add_node(concept, type='concept')
                knowledge_graph.add_edge(video, concept)

        # 创建仅包含概念节点的子图
        concept_graph = knowledge_graph.subgraph(
            [node for node, data in knowledge_graph.nodes(data=True) if data.get('type') == 'concept']
        ).copy()

        return knowledge_graph, concept_graph

    # 使用预计算的最短路径
    def get_shortest_path_length(self, source, target, all_shortest_paths):
        # print("get_shortest_path_length")
        if source in all_shortest_paths and target in all_shortest_paths[source]:
            return all_shortest_paths[source][target]
        else:
            return float('inf')  # 无路径时返回无穷大


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
        student_watch_data_list = []

        scores = {f'hits@{k}': [] for k in k_list}
        scores.update({f'map@{k}': [] for k in k_list})

        # 提取课程对应视频的映射关系
        course_video_mapping = self.build_course_video_mapping(courses)
        # 绘制知识图谱
        graph = ConceptGraph(
            video_concept_file='/kaggle/input/riginmooccube/MOOCCube/relations/video-concept.json',
            parent_son_file='/kaggle/input/riginmooccube/MOOCCube/relations/parent-son.json'
        )
        knowledge_graph, concept_graph = graph.draw_knowledge_graph()
        max_path_length = 2  # 或者设置为2，依赖你的需求
        # print("预先计算所有节点之间的最短路径")
        all_shortest_paths = dict(nx.all_pairs_shortest_path_length(concept_graph, cutoff=max_path_length))
        # print("()预先计算所有节点之间的最短路径")
        grade = 0

        for p_, y_, y_p, wc, dt, wt, d1, d2, d3 in zip(y_prob, y_true, y_prev, w_c, d_t, w_t, d_1, d_2, d_3):
            if y_ == self.PAD:
                student_watch_data_list = []
                grade = 0
                continue

            scores_len += 1
            initial_topk = self.get_top_k_predictions(p_, k=40)
            prev_video_name = idx2u[y_p]
            prev_courses = self.get_courses_by_video(prev_video_name, course_video_mapping)
            prev_course = prev_courses[0]
            student_watch_data_list.append(StudentWatchData(prev_video_name, prev_course, wc, wt, dt))
            # student_watch_data_list.append(prev_video_name)

            # if prev_courses and prev_courses[0] not in prev_course_list:
            #     prev_course_list.insert(0, prev_courses[0])

            # # # ---------------------- 喜好排序
            # score_opt2 = self.optimize_based_on_studentprefer2(student_watch_data_list, graph, knowledge_graph,
            #                                                    initial_topk, idx2u, prev_course,
            #                                                    course_video_mapping,
            #                                                    all_shortest_paths)

            # ---------------------nearby1-4
            scores_pro, f_next_video = self.score_nearby(initial_topk, y_p, idx2u, course_video_mapping, courses,
                                                              prev_courses)

            # ------------------- 概念距离排序0
            focus_concepts = graph.find_focus_concept(prev_video_name)

            if wc > 1 or d2 > 1 :
                score_opt = self.optimize_topk_based_on_concept1(knowledge_graph, focus_concepts, initial_topk, idx2u, graph, all_shortest_paths)
                # sorted_topk = self.reorder_top_predictions(initial_topk, score_opt)
                score = self.merge_scores(score_opt, scores_pro)

            else:
                # sorted_topk = list(initial_topk)
                score = scores_pro
            # 根据得分重新排序topk
            # score = scores_pro
            sorted_topk = self.reorder_top_predictions(initial_topk, score)

            # -------------------单独使用一个分数排序
            # if score_opt2 is not None:
            #     sorted_topk = self.reorder_top_predictions(initial_topk, score_opt2)

            # focus_concepts = graph.find_focus_concept(prev_video_name)
            # if len(student_watch_data_list) < 3 and wc > 1 and d2 > 1:
            #     score_opt = self.optimize_topk_based_on_concept2(knowledge_graph, focus_concepts, initial_topk, idx2u,
            #                                                      graph, all_shortest_paths)
            #     sorted_topk = self.reorder_top_predictions(initial_topk, score_opt)
            # else:
            #     sorted_topk = list(initial_topk)

            # ---------------------如果找到 next_video_id，则将其插入到首位
            next_video_id = self.find_next_video(prev_video_name, prev_course, u2idx, courses)
            if next_video_id is not None and next_video_id not in sorted_topk:
                sorted_topk.insert(0, next_video_id)

            # 更新结果
            for k in k_list:
                topk = sorted_topk[:k]
                scores[f'hits@{k}'].append(1.0 if y_ in topk else 0.0)
                scores[f'map@{k}'].append(self.apk([y_], topk, k))

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len

    def find_next_video(self, prev_video_name, prev_course, u2idx, courses):
        """在课程中找到相邻的下一个视频"""
        # prev_courses = self.get_courses_by_video(prev_video_name, course_video_mapping)

        for course in courses:
            if course['id'] == prev_course:
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

    # def build_course_video_mapping(self, courses):
    #     """构建课程与视频的映射关系，减少重复查找"""
    #     mapping = {}
    #     for course in courses:
    #         course_id = course['id']
    #         video_order = course.get('video_order', [])
    #         mapping[course_id] = video_order
    #     return mapping

    def build_course_video_mapping(self, courses):
        """构建课程与视频的映射关系，减少重复查找"""
        mapping = defaultdict(list)
        for course in courses:
            mapping[course['id']].extend(course.get('video_order', []))
        return dict(mapping)

    def get_top_k_predictions(self, p_, k=20):
        """获取排序后的前K个预测视频"""
        return p_.argsort()[::-1][:k]

    def score_nearby(self, topk, y_p, idx2u, course_video_mapping, courses, prev_courses):
        """根据与历史视频的关联性给每个预测视频评分"""
        scores_pro = {video_id: 0 for video_id in topk}
        # scores_pro = {video_id: (20-i) if i < 20 else 0 for i, video_id in enumerate(topk)}
        prev_video_name = idx2u[y_p]

        for video_id in topk:
            predicted_video_name = idx2u[video_id]
            predicted_courses = self.get_courses_by_video(predicted_video_name, course_video_mapping)

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
                                score = 15  # 确保相邻视频加足够高的分数
                                f_next_video = False  # 标记为不需要再找下一个视频
                            elif distance == -1:
                                score = 14
                            elif distance == 2:
                                score = 13
                            elif distance == -2:
                                score = 12
                            elif distance == 3:
                                score = 11
                            elif distance == -3:
                                score = 10
                            elif distance == 4:
                                score = 9
                            elif distance == -4:
                                score = 8
                            elif distance == 5:
                                score = 7
                            elif distance == 6:
                                score = 6
                            elif distance == 7:
                                score = 5
                        except ValueError:
                            continue

        return score, f_next_video

    def reorder_top_predictions(self, topk, scores_pro):
        """根据得分重新排序topk预测视频"""
        #scores_pro是字典
        scored_videos = sorted(((v, scores_pro[v]) for v in topk if scores_pro[v] > 0), key=lambda x: x[1],
                               reverse=True)
        unscored_videos = [v for v in topk if scores_pro[v] == 0]

        sorted_videos = [v for v, _ in scored_videos] + unscored_videos

        return sorted_videos

    def optimize_topk_based_on_concept1(self, knowledge_graph, focus_concepts, topk, idx2u, graph,
                                        all_shortest_paths):
        # video_scores = {}  # 用于存储视频及其累计相关性得分
        zero_score_videos_set = set()  # 用于去重存储得分为0的视频
        topk_scores = {video_id: (15 - i) if i < 15 else 0 for i, video_id in enumerate(topk)}
        # scores_opt = scores
        scores_opt = {video_id: 0 for video_id in topk}

        for video in topk:
            video_name = idx2u[video]  # 获取视频名称
            if video_name in knowledge_graph:  # 确保视频存在于知识图谱中
                # 获取与视频相关联的概念
                # print("(opt)video_name:", video_name)
                video_concepts = [concept for concept in knowledge_graph.neighbors(video_name) if
                                  concept.startswith('K_')]
                # print("(opt)video_concepts:", video_concepts)

                # 计算相关性得分
                for concept in video_concepts:
                    for focus_concept in focus_concepts:
                        # shortest_path = graph.direct_get_shortest_path_length(concept, focus_concept, concept_graph)
                        shortest_path = graph.get_shortest_path_length(concept, focus_concept, all_shortest_paths)

                        if shortest_path != float('inf'):
                            if shortest_path == 0:
                                scores_opt[video] += 1
                            # if scores_opt[video] == scores[video]:
                            # scores_opt[video] += (1 / (1 + shortest_path))

            # 如果得分为0，将其标记为零分视频
            if scores_opt[video] == 0:
                zero_score_videos_set.add(video)
            else:
                # 如果视频之前在 zero_score_videos_set 中，现在有得分，移除它
                zero_score_videos_set.discard(video)

        # 将有得分的视频按得分排序
        optimized_topk = sorted([(video, score) for video, score in scores_opt.items() if score > 0],
                                key=lambda x: x[1], reverse=True)

        highscore_videos = [video for video, score in optimized_topk if score > 4]
        mediumscore_videos = [video for video, score in optimized_topk if (score > 2 and score < 5)]
        # limited_video_scores = {video: video_scores[video] if video in top5_videos else 0 for video in topk}
        # limited_video_scores = {video: (scores_opt[video]-2) if video in highscore_videos else scores_opt[video] for video in topk}
        limited_video_scores = {
            video: (
                scores_opt[video] - 2 if video in highscore_videos
                else scores_opt[video] - 1 if video in mediumscore_videos
                else scores_opt[video]
            )
            for video in topk
        }
        final_scores = {video: limited_video_scores.get(video, 0) + topk_scores.get(video, 0) for video in topk}
        scores_opt = sorted([(video, score) for video, score in final_scores.items()], key=lambda x: x[1],
                              reverse=True)
        scores_opt = dict(scores_opt)
        # # 提取排序后的视频ID
        # sorted_videos_with_scores = [video for video, score in optimized_topk]
        #
        # # 将得分为0的视频保持原有顺序，追加到排序后的视频ID列表末尾
        # final_topk = sorted_videos_with_scores + list(zero_score_videos_set)

        # return final_topk
        return scores_opt

    def optimize_topk_based_on_concept2(self, knowledge_graph, focus_concepts, topk, idx2u, graph,
                                        all_shortest_paths):

        scores_ori = {video_id: (15 - i) if i < 15 else 0 for i, video_id in enumerate(topk)}
        scores_opt = {video_id: 0 for video_id in topk}

        for video in topk:
            video_name = idx2u[video]  # 获取视频名称
            if video_name in knowledge_graph:  # 确保视频存在于知识图谱中
                # 获取与视频相关联的概念
                # print("(opt)video_name:", video_name)
                video_concepts = [concept for concept in knowledge_graph.neighbors(video_name) if
                                  concept.startswith('K_')]

                # 计算相关性得分
                for concept in video_concepts:
                    for focus_concept in focus_concepts:
                        # shortest_path = graph.direct_get_shortest_path_length(concept, focus_concept, concept_graph)
                        shortest_path = graph.get_shortest_path_length(concept, focus_concept, all_shortest_paths)

                        if shortest_path != float('inf'):
                            if shortest_path == 0:
                                scores_opt[video] += 1
                            # if scores_opt[video] == scores[video]:
                            # scores_opt[video] += (1 / (1 + shortest_path))

        scores_opt = {
            video: scores_opt.get(video, 0) + scores_ori.get(video, 0)
            for video in topk
        }
        # for video, score in scores_opt.items():
        #     print(f"Course: {video}, Score: {score}")
        return scores_opt


    def merge_scores(self, scores_pro1, scores_pro2):
        merged_scores = {}

        # 遍历第一个字典，将所有键值对添加到merged_scores中
        for video_id, score in scores_pro1.items():
            merged_scores[video_id] = score

        # 遍历第二个字典，如果键已经在merged_scores中，则将分数相加，否则直接添加
        for video_id, score in scores_pro2.items():
            if video_id in merged_scores:
                merged_scores[video_id] += score  # 分数相加
            else:
                merged_scores[video_id] = score  # 如果不存在，直接添加

        return merged_scores

    def multiply_scores(self, scores_pro1, scores_pro2):
        merged_scores = {}

        # 遍历第一个字典，将所有键值对添加到merged_scores中
        for video_id, score in scores_pro1.items():
            if video_id in scores_pro2:
                merged_scores[video_id] = score * scores_pro2[video_id]  # 分数相乘
            else:
                merged_scores[video_id] = score  # 如果不存在，直接保留原始分数

        # 遍历第二个字典，处理那些不在第一个字典中的video_id
        for video_id, score in scores_pro2.items():
            if video_id not in merged_scores:
                merged_scores[video_id] = score  # 直接添加不存在的video_id

        for video, score in merged_scores.items():
            print(f"Course: {video}, Score: {score}")

        return merged_scores

    def optimize_based_on_studentprefer(self, StudentWatchData_list, graph, knowledge_graph, topk,
                                        idx2u, prev_course, course_video_mapping, all_shortest_paths):
        # # 初始化视频的匹配分数
        # video_scores = {video_id: 0 for video_id in topk}
        video_scores = {video_id: 50 if i < 10 else 0 for i, video_id in enumerate(topk)}
        score = 1

        # zero_score_videos_set = set()
        # if len(StudentWatchData_list) > 10:
        #     StudentWatchData_list = StudentWatchData_list[-10:]
        reversed_list = StudentWatchData_list[::-1]
        for former_video_name in reversed_list:
            former_courses = self.get_courses_by_video(former_video_name.video_name, course_video_mapping)
            former_course = former_courses[0]
            # if former_course != prev_course and (former_video_name.watch_count > 1 or (former_video_name.watch_time/former_video_name.total_time) > 1):
            if former_course != prev_course:
                focus_concepts =graph.find_focus_concept(former_video_name.video_name)
                for video in topk:
                    video_name = idx2u[video]  # 获取视频名称
                    if video_name in knowledge_graph:  # 确保视频存在于知识图谱中
                        video_concepts = [concept for concept in knowledge_graph.neighbors(video_name) if
                                          concept.startswith('K_')]
                        for concept in video_concepts:
                            for focus_concept in focus_concepts:
                                shortest_path = graph.get_shortest_path_length(concept, focus_concept,
                                                                               all_shortest_paths)
                                if shortest_path == 0 and video_scores[video] != 50:
                                    video_scores[video] += score
                    # if video_scores[video] == 0:
                    #     zero_score_videos_set.add(video)
                    # else:
                    #     # 如果视频之前在 zero_score_videos_set 中，现在有得分，移除它
                    #     zero_score_videos_set.discard(video)

                return video_scores

        return None

    def optimize_based_on_studentprefer2(self, StudentWatchData_list, graph, knowledge_graph, topk,
                                        idx2u, prev_course, course_video_mapping, all_shortest_paths):
        # # 初始化视频的匹配分数
        # video_scores = {video_id: 0 for video_id in topk}
        video_scores = {video_id: 50 if i < 10 else 0 for i, video_id in enumerate(topk)}
        score = 1

        # zero_score_videos_set = set()
        # if len(StudentWatchData_list) > 10:
        #     StudentWatchData_list = StudentWatchData_list[-10:]
        reversed_list = StudentWatchData_list[::-1]
        for former_video_name in reversed_list:
            former_courses = self.get_courses_by_video(former_video_name.video_name, course_video_mapping)
            former_course = former_courses[0]
            # if former_course != prev_course and (former_video_name.watch_count > 1 or (former_video_name.watch_time/former_video_name.total_time) > 1):
            if former_course != prev_course:
                focus_concepts =graph.find_focus_concept(former_video_name.video_name)
                for video in topk:
                    video_name = idx2u[video]  # 获取视频名称
                    if video_name in knowledge_graph:  # 确保视频存在于知识图谱中
                        video_concepts = [concept for concept in knowledge_graph.neighbors(video_name) if
                                          concept.startswith('K_')]
                        for concept in video_concepts:
                            for focus_concept in focus_concepts:
                                shortest_path = graph.get_shortest_path_length(concept, focus_concept,
                                                                               all_shortest_paths)
                                if shortest_path == 0 and video_scores[video]!=50:
                                    video_scores[video] += score

                return video_scores

        return None
