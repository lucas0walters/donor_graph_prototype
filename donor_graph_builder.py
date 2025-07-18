import pandas as pd
import networkx as nx
import pickle

# Load the tab-separated campaign finance data
# This program works with as a prototype using a one-month slice of data of donation data to Florida candidates
class DonorGraphBuilder:
    def __init__(self, file_path):
        self.file_path = file_path
        self.log = ["Initialized"]

    def __str__(self):
        return "\n".join(self.log)

    def build_graph(self):
        df = pd.read_csv(self.file_path, sep="\t")

        # Clean and normalize contributor and recipient fields
        df['Contributor'] = df['Contributor Name'].str.upper().str.strip()
        df['Recipient'] = df['Candidate/Committee'].str.upper().str.strip()

        # Group data to aggregate donation amounts and frequency
        grouped = df.groupby(['Contributor', 'Recipient']).agg(
            total_amount=pd.NamedAgg(column='Amount', aggfunc='sum'),
            frequency=pd.NamedAgg(column='Amount', aggfunc='count')
        ).reset_index()

        # Create bipartite graph
        G = nx.Graph()

        # Add nodes with bipartite attribute
        contributors = set(grouped['Contributor'])
        recipients = set(grouped['Recipient'])
        G.add_nodes_from(contributors, bipartite=0)
        G.add_nodes_from(recipients, bipartite=1)

        # Add edges with aggregated weights
        for _, row in grouped.iterrows():
            normalized_amount = row['total_amount'] / grouped['total_amount'].sum()
            G.add_edge(
                row['Contributor'],
                row['Recipient'],
                weight_amount=normalized_amount,
                weight_count=row['frequency']
            )

        self.log.append(f"Graph created and returned with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

if __name__ == "__main__":
    builder = DonorGraphBuilder("onemonthslice.txt")
    graph = builder.build_graph()
    print(builder)