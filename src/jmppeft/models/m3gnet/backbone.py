import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data.data import BaseData
from torch_scatter import scatter_sum
from .modules.layers import MLP, GatedMLP
from .modules.angle_encoder import SphericalBasisLayer
from .modules.edge_encoder import SmoothBesselBasis
from .modules.message_passing import MainBlock
from jmppeft.utils.goc_graph import graphs_from_batch
from typing import TypedDict


class M3GNetBackboneOutput(TypedDict):
    node_attr: torch.Tensor
    edge_attr: torch.Tensor
    V_st: torch.Tensor
    D_st: torch.Tensor
    idx_s: torch.Tensor
    idx_t: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor
    

class M3GNet(nn.Module):
    """
    M3GNet Implemented with Pytorch
    Paper Reference: https://arxiv.org/pdf/2202.02450.pdf
    """
    def __init__(
        self,
        num_layers: int = 4,
        hidden_dim: int = 64,
        max_l: int = 4,
        max_n: int = 4,
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        max_z: int = 94, ## Max Number of Elements
        **kwargs,
    ):
        super(M3GNet, self).__init__()
        self.name = "M3GNet"
        self.rbf = SmoothBesselBasis(r_max=cutoff, max_n=max_n)
        self.sbf = SphericalBasisLayer(max_n=max_n, max_l=max_l, cutoff=cutoff)
        self.edge_encoder = MLP(in_dim=max_n, out_dims=[hidden_dim], activation="swish", use_bias=False)
        module_list = [MainBlock(cutoff, threebody_cutoff, hidden_dim, max_n, max_l) for _ in range(num_layers)]
        self.main_blocks = nn.ModuleList(module_list)
        self.energy_final = GatedMLP(in_dim=hidden_dim, out_dims=[hidden_dim, hidden_dim, hidden_dim], activation=["swish", "swish", None])
        self.force_final = GatedMLP(in_dim=hidden_dim, out_dims=[hidden_dim, hidden_dim, hidden_dim], activation=["swish", "swish", None])
        self.apply(self.init_weights)
        self.atom_embedding = MLP(in_dim=max_z + 1, out_dims=[hidden_dim], activation=None, use_bias=False)
        self.atom_embedding.apply(self.init_weights_uniform)
        self.max_z = max_z
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_l = max_l
        self.max_n = max_n
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        
    def forward(
        self,
        data: BaseData,
    ):        
        atomic_numbers = data.atomic_numbers
        edge_index = getattr(data, "a2a_edge_index")
        edge_length = getattr(data, "a2a_distance")
        edge_vector = getattr(data, "a2a_vector")
        three_body_indices = getattr(data, "a2ee2a_edge_index")
        triple_edge_length = getattr(data, "a2ee2a_distance")
        num_triple_ij = torch.bincount(three_body_indices[0], minlength=edge_index.shape[1])
        vij = edge_vector[three_body_indices[0].clone()]
        vik = edge_vector[three_body_indices[1].clone()]
        rij = edge_length[three_body_indices[0].clone()]
        rik = edge_length[three_body_indices[1].clone()]
        cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik)
        cos_jik = torch.clamp(cos_jik, min=-1. + 1e-7, max=1.0 - 1e-7)
        triple_edge_length = rik.view(-1)
        edge_length = edge_length.unsqueeze(-1)
        

        ## featurize
        atom_attr = self.atom_embedding(self.one_hot_atoms(atomic_numbers))
        edge_attr = self.rbf(edge_length.view(-1))
        edge_attr_zero = edge_attr  ## e_ij^0
        edge_attr = self.edge_encoder(edge_attr)
        three_basis = self.sbf(triple_edge_length, torch.acos(cos_jik))

        # Main Loop
        for _, main_block in enumerate(self.main_blocks):
            atom_attr, edge_attr = main_block(
                atom_attr,
                edge_attr,
                edge_attr_zero,
                edge_index,
                three_basis,
                three_body_indices,
                edge_length,
                num_triple_ij,
            )

        X_e = self.energy_final(atom_attr)
        # energies = scatter_sum(energies_i, batch_idx, dim=0, dim_size=num_graphs)

        X_f = self.force_final(edge_attr)
        # forces_ij = forces_ij * edge_vector / edge_length.view(-1, 1)
        # forces_i = scatter_sum(forces_ij, edge_index[1], dim=0, dim_size=num_atoms)
        
        out: M3GNetBackboneOutput = {
            "node_attr": atom_attr,
            "edge_attr": edge_attr,
            "V_st": edge_vector,
            "D_st": edge_length,
            "idx_s": edge_index[0],
            "idx_t": edge_index[1],
            "energy": X_e,
            "forces": X_f,
        }
        return out
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def init_weights_uniform(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, a=-0.05, b=0.05)
            
    def one_hot_atoms(self, species):
        return F.one_hot(species, num_classes=self.max_z + 1).float()
    
    def get_model_params(self):
        return {
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "max_l": self.max_l,
            "max_n": self.max_n,
            "cutoff": self.cutoff,
            "threebody_cutoff": self.threebody_cutoff,
            "max_z": self.max_z,
        }
    
    def save(
        self,
        path: str,
    ):
        model_dict = {
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "max_l": self.max_l,
            "max_n": self.max_n,
            "cutoff": self.cutoff,
            "threebody_cutoff": self.threebody_cutoff,
            "max_z": self.max_z,
            "state_dict": self.state_dict(),
        }
        torch.save(model_dict, path)
        
    @staticmethod
    def load(
        path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        model_dict = torch.load(path)
        model = M3GNet(
            num_layers=model_dict["num_layers"],
            hidden_dim=model_dict["hidden_dim"],
            max_l=model_dict["max_l"],
            max_n=model_dict["max_n"],
            cutoff=model_dict["cutoff"],
            threebody_cutoff=model_dict["threebody_cutoff"],
            max_z=model_dict["max_z"],
        )
        model.load_state_dict(model_dict["state_dict"])
        return model