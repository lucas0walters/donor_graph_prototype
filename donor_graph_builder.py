import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import faiss
from sklearn.preprocessing import normalize

# Load the tab-separated campaign finance data
# This program works with as a prototype using a one-month slice of data of donation data to Florida candidates
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

        # Add nodes with bipartite attribute
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
    
    def cosine_clustering(self, vectors):
        self.log("Normalizing vectors for cosine similarity...")
        normalized_vectors = normalize(vectors, axis=1)

        self.log("Building FAISS index for clustering...")
        index = faiss.IndexFlatIP(normalized_vectors.shape[1])
        index.add(normalized_vectors.astype('float32'))

        self.log("Clustering vectors using FAISS...")
        distances, indices = index.search(normalized_vectors.astype('float32'), k=10)
        return distances, indices

if __name__ == "__main__":
    builder = DonorGraphBuilder("onemonthslice.txt")
    graph = builder.build_graph() # ADJUST WORKER THREADS ACCORDING TO YOUR SYSTEM! -1 + (number of logical processors)
    vectors = builder.graph_to_vectors(graph)
    print(vectors.head())
    distances, clusters = builder.cosine_clustering(vectors)
    # pd.DataFrame(clusters).to_csv("clusters_output.csv", index=False)
    # print("Clusters exported to clusters_output.csv")