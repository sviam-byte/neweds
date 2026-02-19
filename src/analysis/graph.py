"""Модуль анализа топологии графов.

Преобразует матрицы связности в интерпретируемые метрики Network Science:
центральности, локальная кластеризация, сообщества и глобальные свойства графа.
"""

from __future__ import annotations

from typing import Any, Dict

import networkx as nx
import numpy as np
import pandas as pd


def analyze_graph_topology(
    matrix: np.ndarray,
    node_names: list[str],
    threshold: float = 0.2,
    directed: bool = False,
) -> Dict[str, Any]:
    """Рассчитывает базовые топологические метрики по матрице связности.

    Args:
        matrix: Квадратная матрица связности ``(N x N)``.
        node_names: Имена узлов длиной ``N``.
        threshold: Минимальная сила ребра (по ``abs(weight)``) для включения в граф.
        directed: Строить направленный граф.

    Returns:
        Словарь с ключами:
        - ``node_metrics``: таблица метрик по узлам;
        - ``global_metrics``: глобальные свойства графа;
        - ``graph_obj``: объект ``networkx`` для дальнейшей визуализации.

        Если после порога ребер не осталось — возвращает ``{"error": ...}``.
    """
    n = int(matrix.shape[0])
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return {"error": "Ожидается квадратная матрица связности"}

    if len(node_names) != n:
        node_names = [str(name) for name in node_names[:n]] + [f"node_{i}" for i in range(len(node_names), n)]

    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            weight = float(matrix[i, j])
            if np.isnan(weight):
                continue
            if abs(weight) >= float(threshold):
                graph.add_edge(i, j, weight=abs(weight), value=weight)

    if graph.number_of_edges() == 0:
        return {"error": "Граф пуст (слишком высокий порог)"}

    degree = dict(graph.in_degree(weight="weight")) if directed else dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, weight="weight")

    try:
        eigenvector = nx.eigenvector_centrality(graph, weight="weight", max_iter=500)
    except Exception:
        eigenvector = {i: 0.0 for i in range(n)}

    clustering = nx.clustering(graph, weight="weight")

    df_nodes = pd.DataFrame(
        {
            "Name": node_names,
            "Degree": [degree.get(i, 0.0) for i in range(n)],
            "Betweenness": [betweenness.get(i, 0.0) for i in range(n)],
            "Eigenvector": [eigenvector.get(i, 0.0) for i in range(n)],
            "Clustering": [clustering.get(i, 0.0) for i in range(n)],
        }
    ).sort_values("Degree", ascending=False)

    num_communities = 0
    try:
        communities = (
            nx.community.greedy_modularity_communities(graph.to_undirected())
            if directed
            else nx.community.greedy_modularity_communities(graph)
        )
        comm_map: dict[int, int] = {}
        for idx, comm in enumerate(communities):
            for node_idx in comm:
                comm_map[int(node_idx)] = idx

        df_nodes["Community"] = [comm_map.get(i, -1) for i in range(n)]
        num_communities = len(communities)
    except Exception:
        pass

    global_metrics = {
        "density": nx.density(graph),
        "avg_clustering": nx.average_clustering(graph, weight="weight"),
        "communities_count": num_communities,
        "is_connected": nx.is_strongly_connected(graph) if directed else nx.is_connected(graph),
    }

    return {
        "node_metrics": df_nodes,
        "global_metrics": global_metrics,
        "graph_obj": graph,
    }
