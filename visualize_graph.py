import os
import pickle
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_and_visualize_graph():
    # 读取pickle文件
    pickle_path = '/Users/ericyoung/ysx/code/github-study/RAG/hippoRag/outputs/qwen3-4b_text-embedding-bge-m3/graph.pickle'
    try:
        # 检查文件是否存在
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f'找不到图谱文件：{pickle_path}')
            
        print(f'正在读取图谱文件：{pickle_path}')
        with open(pickle_path, 'rb') as f:
            graph = pickle.load(f)
            
        # 打印图谱基本信息
        print(f'成功加载图谱，节点数：{len(graph.vs)}，边数：{len(graph.es)}')
        print('图谱类型：', type(graph))
        
        # 从igraph转换为NetworkX图对象
        G = nx.Graph()
        # 首先添加所有节点
        for i in range(len(graph.vs)):
            G.add_node(i)
        # 然后添加边
        for edge in graph.es:
            G.add_edge(edge.source, edge.target)
        print(f'NetworkX图对象创建成功，节点数：{G.number_of_nodes()}，边数：{G.number_of_edges()}')
        
        # 获取节点标签
        node_labels = {}
        try:
            for i, vertex in enumerate(graph.vs):
                content = vertex['content'] if 'content' in vertex.attributes() else str(i)
                # 如果内容太长，只取前10个字符
                node_labels[i] = str(content)[:20] + '...' if len(str(content)) > 20 else str(content)
            print('成功获取节点标签')
        except Exception as e:
            print(f'获取节点标签时出错：{str(e)}')
            # 使用节点索引作为默认标签
            node_labels = {i: str(i) for i in range(len(graph.vs))}
        
        # 设置图形大小和布局参数
        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        ax = fig.add_subplot(111)
        
        # 设置布局
        import numpy as np
        np.random.seed(42)  # 使用固定的随机种子
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # 绘制图形
        nx.draw(G, pos, ax=ax, with_labels=True, labels=node_labels,
                node_color='lightblue', node_size=2000, font_size=8,
                font_weight='bold', edge_color='gray', width=1)
        
        # 添加标题
        ax.set_title('知识图谱可视化', fontsize=16, pad=20)
        
        # 生成带时间戳的文件名
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(os.path.dirname(pickle_path), f'knowledge_graph_{timestamp}.png')
        
        # 保存图形
        print(f'正在保存图形到：{save_path}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print('图形保存成功')
        
        # 显示图形
        plt.show()
        
    except Exception as e:
        print(f'读取或可视化图形时发生错误：{str(e)}')

if __name__ == '__main__':
    load_and_visualize_graph()