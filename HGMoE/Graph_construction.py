import torch
from torchvision.models import resnet50
from torchvision import transforms
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib

model = resnet50(pretrained=True)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def generate_adjacency_matrix(edge_index, num_nodes):
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))
    for edge in edge_index.t():
        i, j = edge
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    return adjacency_matrix


def generate_weighted_adjacency_matrix(edge_index, edge_weight, num_nodes):
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))
    for idx in range(edge_index.size(1)):
        u, v = edge_index[0][idx].item(), edge_index[1][idx].item()
        weight = edge_weight[idx].item()
        adjacency_matrix[u][v] = weight
        adjacency_matrix[v][u] = weight
    return adjacency_matrix


def construct_graph(features, category_labels, temporal_weight=1.0, self_loop_weight=1.0, threshold=0.95):
    G = nx.Graph()
    for idx, feature in enumerate(features):
        G.add_node(idx)
    for i in range(len(features) - 1):
        G.add_edge(i, i + 1, weight=temporal_weight, type='temporal')
    for idx in G.nodes():
        G.add_edge(idx, idx, weight=self_loop_weight, type='self-loop')
    semantic_edges = compute_semantic_edges(features, category_labels, threshold)
    for edge in semantic_edges:
        node1, node2, weight = edge
        G.add_edge(node1, node2, weight=weight, type='semantic')
    return G


def compute_semantic_edges(features, category_labels, threshold):
    semantic_edges = []
    num_frames = len(features)
    flattened_features = [f.flatten() for f in features]
    feature_similarities = cosine_similarity(flattened_features)
    category_similarities = np.zeros((num_frames, num_frames))
    for i in range(num_frames):
        for j in range(num_frames):
            if category_labels[i] == category_labels[j]:
                category_similarities[i, j] = 1
    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            semantic_weight = feature_similarities[i, j]
            if semantic_weight > threshold:
                semantic_edges.append((i, j, semantic_weight))
    return semantic_edges
