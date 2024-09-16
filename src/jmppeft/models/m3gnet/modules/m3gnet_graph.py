from torch_geometric.data.data import BaseData
from torch_geometric.data.batch import Batch

def graph_from_data(data:BaseData|Batch, cutoff:float, threebody_cutoff:float):
    """
    Convert a BaseData or Batch object to a graph object
    """
    