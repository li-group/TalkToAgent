import json

with open('./dpo_prompt(part).json', 'r', encoding='utf-8') as f:
    dpo_prompt = json.load(f)
with open('./prompt_56_0708(part).json', 'r', encoding='utf-8') as f:
    dpo_prompt2 = json.load(f)





# %% import networkx as nx
import matplotlib.pyplot as plt
import numpy as np  # 가중치 스케일링을 위해 numpy를 사용할 수 있습니다.


def draw_weighted_directed_graph(transition_matrix_data, title="Directed Graph with Weighted Edges"):
    """
    전이 행렬 데이터를 기반으로 가중치가 적용된 방향성 그래프를 그립니다.

    Args:
        transition_matrix_data (dict): 엣지 가중치를 나타내는 딕셔너리.
                                       예: {('A', 'B'): 0.7, ('B', 'A'): 0.3, ('A', 'A'): 0.5}
        title (str): 그래프의 제목.
    """
    G = nx.DiGraph()

    # 노드와 엣지 추가, 그리고 엣지 가중치 설정
    edge_weights = []
    for (source, target), weight in transition_matrix_data.items():
        G.add_edge(source, target, weight=weight)
        edge_weights.append(weight)

    # 엣지 두께 스케일링 (최소/최대 두께를 설정하여 시각적 구분을 명확히 할 수 있습니다)
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)

        # 엣지 두께의 최소값과 최대값을 설정하여 너무 얇거나 두껍지 않게 조절
        min_line_width = 1.0
        max_line_width = 10.0

        # 가중치를 라인 두께로 매핑
        scaled_widths = {}
        for (source, target), weight in transition_matrix_data.items():
            if max_weight == min_weight:  # 모든 가중치가 동일한 경우
                scaled_widths[(source, target)] = (min_line_width + max_line_width) / 2
            else:
                # 선형 스케일링
                scaled_widths[(source, target)] = min_line_width + \
                                                  (max_line_width - min_line_width) * \
                                                  ((weight - min_weight) / (max_weight - min_weight))
    else:
        scaled_widths = {}

    # 노드 레이아웃 설정 (원형 레이아웃이 self-loop와 사이클을 보기 좋습니다)
    pos = nx.circular_layout(G)
    # pos = nx.spring_layout(G, k=0.5, iterations=50) # 다른 레이아웃 옵션

    plt.figure(figsize=(10, 8))

    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=3000, edgecolors='black', linewidths=1.5)

    # 일반 엣지 및 self-loop를 분리하여 그리기
    # self-loop를 제외한 엣지 리스트
    normal_edges = [(u, v) for u, v in G.edges() if u != v]
    # self-loop 엣지 리스트
    self_loops = [(u, v) for u, v in G.edges() if u == v]

    # 일반 엣지 그리기
    if normal_edges:
        nx.draw_networkx_edges(G, pos,
                               edgelist=normal_edges,
                               width=[scaled_widths[(u, v)] for u, v in normal_edges],
                               arrowstyle='->', arrowsize=20, edge_color='gray',
                               alpha=0.7, connectionstyle="arc3,rad=0.1")  # 약간의 곡률을 주어 겹치지 않게

    # self-loop 그리기 (별도의 connectionstyle로 원형 표현)
    if self_loops:
        for u, v in self_loops:
            nx.draw_networkx_edges(G, pos,
                                   edgelist=[(u, v)],
                                   width=scaled_widths[(u, v)],
                                   arrowstyle='->', arrowsize=20, edge_color='red',  # self-loop는 빨간색으로 구분 (선택 사항)
                                   connectionstyle="arc3,rad=100.2",  # self-loop의 곡률 조절
                                   alpha=0.8)

    # 노드 라벨 그리기
    nx.draw_networkx_labels(G, pos, font_size=15, font_weight='bold', font_color='black')

    # 엣지 가중치(라벨) 그리기 (선택 사항: 엣지 위에 숫자를 표시하고 싶다면)
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title(title, size=18)
    plt.axis('off')  # 축 숨기기
    plt.show()


# --- 예시 사용법 ---
if __name__ == "__main__":
    # 전이 행렬 데이터 예시
    # 키는 (소스 노드, 타겟 노드) 튜플, 값은 전이 빈도/가중치
    example_transition_matrix = {
        ('A', 'B'): 0.7,
        ('B', 'C'): 0.6,
        ('C', 'A'): 0.8,  # A-B-C-A 사이클
        ('A', 'A'): 10.5,  # A에 대한 Self-loop
        ('C', 'D'): 0.4,
        ('D', 'A'): 0.9,
        ('B', 'B'): 0.3,  # B에 대한 Self-loop
        ('D', 'D'): 0.2,  # D에 대한 Self-loop
        ('D', 'B'): 0.1,
        ('B', 'D'): 0.25  # 새로운 엣지 추가
    }

    draw_weighted_directed_graph(example_transition_matrix, "Transition Graph with Weighted Edges and Self-Loops")

    # 또 다른 예시: 단순한 그래프
    simple_transition_matrix = {
        ('X', 'Y'): 10,
        ('Y', 'Z'): 5,
        ('Z', 'X'): 8,
        ('X', 'X'): 3  # self-loop
    }
    # draw_weighted_directed_graph(simple_transition_matrix, "Simple Weighted Graph")