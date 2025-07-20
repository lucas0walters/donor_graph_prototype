import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import faiss
from sklearn.preprocessing import normalize
import numpy as np
import argparse

class DonorGraphBuilder:
    def __init__(self, file_path):
        self.file_path = file_path
        self.status = "Initialized"
        self.log(self.status)

    def __str__(self):
        return self.status

    def log(self, message):
        self.status = message
        print(self.status)

    def build_graph(self):
        self.log("Loading data...")
        df = pd.read_csv(self.file_path, sep="\t")

        self.log("Cleaning and normalizing fields...")
        df['Contributor'] = df['Contributor Name'].str.upper().str.strip()
        df['Recipient'] = df['Candidate/Committee'].str.upper().str.strip()

        self.log("Grouping data and aggregating donations...")
        grouped = df.groupby(['Contributor', 'Recipient']).agg(
            total_amount=pd.NamedAgg(column='Amount', aggfunc='sum'),
            frequency=pd.NamedAgg(column='Amount', aggfunc='count')
        ).reset_index()

        self.log("Creating bipartite graph...")
        G = nx.Graph()

        contributors = set(grouped['Contributor'])
        recipients = set(grouped['Recipient'])
        G.add_nodes_from(contributors, bipartite=0)
        G.add_nodes_from(recipients, bipartite=1)

        self.log("Adding edges with weights...")
        for _, row in grouped.iterrows():
            normalized_amount = row['total_amount'] / grouped['total_amount'].sum()
            G.add_edge(
                row['Contributor'],
                row['Recipient'],
                weight_amount=normalized_amount,
                weight_count=row['frequency']
            )

        self.log(f"Graph created and returned with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def graph_to_vectors(self, graph, dimensions=64, walk_length=15, num_walks=30, workers=7):
        self.log("Generating Node2Vec vectors...")
        node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        model = node2vec.fit(window=10, min_count=1)
        vectors = pd.DataFrame(
            [model.wv[str(node)] for node in graph.nodes()],
            index=[str(node) for node in graph.nodes()]
        )
        self.log(f"Node2Vec vectors generated for {len(vectors)} nodes.")
        return vectors
    
    def cosine_clustering(self, vectors, k=10):
        self.log("Normalizing vectors for cosine similarity...")
        normalized_vectors = normalize(vectors, axis=1)

        self.log("Building FAISS index for clustering...")
        index = faiss.IndexFlatIP(normalized_vectors.shape[1])
        index.add(normalized_vectors.astype('float32'))

        self.log("Clustering vectors using FAISS...")
        distances, indices = index.search(normalized_vectors.astype('float32'), k)
        self.log("Clustering completed.")
        return distances, indices
    
    def recommend_by_name(self, graph, vectors, distances, clusters, name):
        donor_nodes = [node for node, data in graph.nodes(data=True) if data.get('bipartite') == 0]

        matches = [i for i, node in enumerate(vectors.index) if name.lower() in node.lower() and node in donor_nodes]
        if not matches:
            self.log(f"No donor nodes containing '{name}' found in the graph.")
            return None

        recommendations = []
        for idx in matches:
            recommended_indices = clusters[idx]
            similarity_scores = distances[idx]
            recommended_names = [vectors.index[i] for i in recommended_indices if vectors.index[i] in donor_nodes]
            recommended_scores = [similarity_scores[j] for j, i in enumerate(recommended_indices) if vectors.index[i] in donor_nodes]
            for rec_name, score in zip(recommended_names, recommended_scores):
                recommendations.append({'searched_name': vectors.index[idx], 'name': rec_name, 'similarity_score': score})

        recommendations_df = pd.DataFrame(recommendations)
        self.log(f"Donor recommendations for nodes containing '{name}' retrieved.")
        return recommendations_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Donor graph builder and recommender")
    parser.add_argument('--mode', choices=['build', 'load'], required=True, help="Mode: 'build' to process raw data from scratch, 'load' to use previous outputs")
    parser.add_argument('--data', type=str, default="onemonthslice.txt", help="Input data file for build mode. Default is 'onemonthslice.txt'.")
    parser.add_argument('--vectors', type=str, default="vectors_output.csv", help="Vectors file for load mode. Default is 'vectors_output.csv'.")
    parser.add_argument('--clusters', type=str, default="clusters_output.csv", help="Clusters file for load mode. Default is 'clusters_output.csv'.")
    parser.add_argument('--distances', type=str, default="distances_output.csv", help="Distances file for load mode. Default is 'distances_output.csv'.")
    parser.add_argument('--search', type=str, default="HONEST LEADERSHIP", help="Search string for recommendations. Default is 'HONEST LEADERSHIP'.")
    parser.add_argument('--threads', type=int, default=7, help="Number of worker threads for Node2Vec. Should match your logical processors -1. Default is 7.")
    args = parser.parse_args()

    if args.mode == 'build':
        builder = DonorGraphBuilder(args.data)
        graph = builder.build_graph()
        vectors = builder.graph_to_vectors(graph, workers=args.threads)
        distances, clusters = builder.cosine_clustering(vectors)
        vectors.to_csv(args.vectors)
        np.savetxt(args.clusters, clusters, delimiter=",", fmt="%d")
        np.savetxt(args.distances, distances, delimiter=",")
        print("Graph, vectors, clusters, and distances exported.")
    elif args.mode == 'load':
        vectors = pd.read_csv(args.vectors, index_col=0)
        clusters = np.loadtxt(args.clusters, delimiter=",", dtype=int)
        distances = np.loadtxt(args.distances, delimiter=",")
        builder = DonorGraphBuilder(args.data)
        graph = builder.build_graph()
    else:
        raise ValueError("Invalid mode selected.")

    recommendations = builder.recommend_by_name(graph, vectors, distances, clusters, args.search)
    print(recommendations)