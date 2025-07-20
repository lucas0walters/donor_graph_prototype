# Final Project for LIS4930
This project analyzes campaign finance data from the Florida Department of State, Division of Elections [Campaign Finance Database](https://dos.elections.myflorida.com/campaign-finance/contributions/). The sample data provided in "onemonthslice.txt" is a one month slice of all contributions made to political candidates in the state of Florida, taken from this database.
The tool uses networkx to build a graph of financial transaction totals between donor and candidate nodes over a one month slice of sample data. This graph is then used to produce vector embeddings using node2vec that characterize deep relationships in the data. These embeddings are than indexed using Facebook AI Similarity Search to perform cosine similarity clustering around each donor in our graph. Finally, these clusters can be used to produce recommendations of donors with a high similarity to a given target donor.
This is a rough prototype based loosely on this [article](https://medium.com/p/cd6d0fc22eb4), for the purpose of demonstrating feasability of this technology for the purpose of donor acquisition. Future versions may rework logic and connect to a vector database.

## Usage
This package uses miniconda for version and dependency management. After cloning the repo, cd to it in the conda shell and:
```
conda env create -f node2vec-env.yml
conda activate node2vec-env
python donor_graph_builder.py --help
python donor_graph_builder.py --mode build
```
All rights reserved