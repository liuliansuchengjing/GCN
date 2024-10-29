import networkx as nx
import matplotlib.pyplot as plt


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

    # def draw_knowledge_graph(self):
    #     """绘制知识图谱"""
    #     # 创建一个有向图
    #     # knowledge_graph = nx.DiGraph()
    #     knowledge_graph = nx.Graph()
    #
    #     # 读取parent-son文件，添加父子概念关系
    #     with open(self.parent_son_file, 'r', encoding='utf-8') as file:
    #         for line in file:
    #             parent, child = line.strip().split('\t')
    #             knowledge_graph.add_edge(parent, child)
    #
    #     # 添加视频与概念的关系
    #     for video, concepts in self.video_concept_mapping.items():
    #         if video not in knowledge_graph:
    #             knowledge_graph.add_node(video, type='video')
    #         for concept in concepts:
    #             if concept not in knowledge_graph:
    #                 knowledge_graph.add_node(concept, type='concept')
    #             knowledge_graph.add_edge(video, concept)
    #
    #     return knowledge_graph
    def draw_knowledge_graph(self):
        # 使用有向图存储知识图谱
        knowledge_graph = nx.DiGraph()

        # 将父子概念关系添加为无向边
        with open(self.parent_son_file, 'r', encoding='utf-8') as file:
            for line in file:
                parent, child = line.strip().split('\t')
                knowledge_graph.add_edge(parent, child)  # 默认有向
                knowledge_graph.add_edge(child, parent)  # 添加逆向边

        # 添加视频与概念的关系
        for video, concepts in self.video_concept_mapping.items():
            if video not in knowledge_graph:
                knowledge_graph.add_node(video, type='video')
            for concept in concepts:
                if concept not in knowledge_graph:
                    knowledge_graph.add_node(concept, type='concept')
                knowledge_graph.add_edge(video, concept)  # 视频到概念有向

        return knowledge_graph

    # 使用预计算的最短路径
    def get_shortest_path_length(self, source, target, all_shortest_paths):
        if source in all_shortest_paths and target in all_shortest_paths[source]:
            return all_shortest_paths[source][target]
        else:
            return float('inf')  # 无路径时返回无穷大


graph = ConceptGraph(
            video_concept_file='E:\\SOFT\\pycharm\\MOOC\\MOOCCube\\MOOCCube\\relations\\video-concept.json',
            parent_son_file='E:\\SOFT\\pycharm\\MOOC\\MOOCCube\\MOOCCube\\relations\\parent-son.json'
        )

knowledge_graph = graph.draw_knowledge_graph()

# 预先计算所有节点之间的最短路径
all_shortest_paths = dict(nx.all_pairs_shortest_path_length(knowledge_graph))
concept1 = 'K_草原动物_地理学'
concept2 = 'K_苔原_地理学'
shortest_path = graph.get_shortest_path_length(concept1, concept2, all_shortest_paths)
print(f"distance between {concept2} and {concept1}: {shortest_path} ")
