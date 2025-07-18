import pandas as pd
import networkx as nx
import pickle

# Load the tab-separated campaign finance data
# This program works with as a prototype using a one-month slice of data of donation data to Florida candidates
file_path = "onemonthslice.txt"
df = pd.read_csv(file_path, sep="\t")

# Clean and normalize contributor and recipient fields
df['Contributor'] = df['Contributor Name'].str.upper().str.strip()
df['Recipient'] = df['Candidate/Committee'].str.upper().str.strip()

# Group data to aggregate donation amounts and frequency
grouped = df.groupby(['Contributor', 'Recipient']).agg(
    total_amount=pd.NamedAgg(column='Amount', aggfunc='sum'),
    frequency=pd.NamedAgg(column='Amount', aggfunc='count')
).reset_index()

# Create directed graph
G = nx.DiGraph()

# Add edges with aggregated weights
for _, row in grouped.iterrows():
    normalized_amount = row['total_amount'] / grouped['total_amount'].sum()
    G.add_edge(
        row['Contributor'],
        row['Recipient'],
        weight_amount=normalized_amount,
        weight_count=row['frequency']
    )

# Graph summary output
print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
# Save the graph using pickle just in case we want to load it later
with open("donor_graph.gpickle", "wb") as f:
    pickle.dump(G, f)
print("Graph exported to donor_graph.gpickle")