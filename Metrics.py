def optimize_topk_based_on_concept(self, knowledge_graph, focus_concepts, sorted_topk, idx2u):
    optimized_topk_list = []
    zero_score_videos = []  # 用来存储得分为0的视频

    for video in sorted_topk:
        video_name = idx2u[video]  # 获取视频名称
        if video_name in knowledge_graph:  # 确保视频存在于知识图谱中
            # 获取与视频相关联的概念
            video_concepts = [concept for concept in knowledge_graph.neighbors(video_name) if concept.startswith('K_')]
            relevance_score = 0  # 初始相关性得分为0
            # 计算相关性得分
            for concept in video_concepts:
                for focus_concept in focus_concepts:
                    try:
                        # 计算概念间的最短路径
                        shortest_path = nx.shortest_path_length(knowledge_graph, source=concept, target=focus_concept)
                        relevance_score += 1 / (1 + shortest_path)
                    except nx.NetworkXNoPath:
                        relevance_score += 0

            # 如果相关性得分为0，添加到zero_score_videos列表
            if relevance_score == 0:
                zero_score_videos.append(video)
            else:
                # 如果有得分，存储为字典
                video_dict = {'video_id': video, 'relevance_score': relevance_score}
                optimized_topk_list.append(video_dict)
        else:
            # 如果视频不在知识图谱中，也视为得分为0
            zero_score_videos.append(video)

    # 对有得分的视频按照相关性得分从高到低进行排序
    optimized_topk = sorted(optimized_topk_list, key=lambda x: x['relevance_score'], reverse=True)

    # 提取排序后的视频ID
    sorted_videos_with_scores = [video_dict['video_id'] for video_dict in optimized_topk]

    # 将得分为0的视频保持原有顺序，追加到排序后的视频ID列表末尾
    final_topk = sorted_videos_with_scores + zero_score_videos

    return final_topk
