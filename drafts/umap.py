from ase import Atoms
from ase.io import read
from dscribe.descriptors import SOAP
from typing import List
import numpy as np
from tqdm import tqdm
import argparse
import os


def main(args_dict: dict):
    species = args_dict["species"]
    model_name = args_dict["model_name"]
    save_dir = "./{}-{}-umap".format("".join(species), model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    ## ================== Calculate SOAP Features ==================
    soap_descriptor = SOAP(
        species=species,
        r_cut=args_dict["r_cut"],         
        n_max=args_dict["n_max"],
        l_max=args_dict["l_max"],    
        periodic=True,
        dtype="float32",
    )
    energy_atoms_list_path = "./temp_data/{}-mptrj.xyz".format("".join(species))
    energy_atoms_list: List[Atoms] = read(energy_atoms_list_path, index=":")
    random_reorder = np.random.permutation(len(energy_atoms_list))
    energy_atoms_list = [energy_atoms_list[i] for i in random_reorder]
    pbar = tqdm(energy_atoms_list, desc="Generating SOAP Features")
    soap_features = {key: [] for key in species}
    energy_per_atom = {key: [] for key in species}
    for i, atoms in enumerate(energy_atoms_list):
        chemical_symbols = atoms.get_chemical_symbols()
        energy_per_atom_i = energy_atoms_list[i].get_potential_energy() / len(chemical_symbols)
        soap_features_i = soap_descriptor.create(atoms, n_jobs=12)
        for j in range(len(soap_features_i)):
            soap_features[chemical_symbols[j]].append(soap_features_i[j])
            energy_per_atom[chemical_symbols[j]].append(energy_per_atom_i)
        pbar.update(1)
        if np.min([len(value) for key, value in soap_features.items()]) > args_dict["min_num_points"]:
            break
    pbar.close()
    soap_features = {key: np.array(value) for key, value in soap_features.items()}
    energy_per_atom = {key: np.array(value) for key, value in energy_per_atom.items()}
    
    ## ================== Load SOAP Features ==================
    all_files = os.listdir("./temp_data")
    all_files = [file for file in all_files if model_name in file and "".join(species) in file]
    all_files = sorted(all_files)
    for i, file in enumerate(all_files):
        num_block = i + 1
        atoms_list_path = "{}-block{}-{}-mptrj.xyz".format(model_name, num_block, "".join(species))
        assert file == atoms_list_path.split("/")[-1], f"{file} != {atoms_list_path.split('/')[-1]}"
        atoms_list: List[Atoms] = read(os.path.join("./temp_data", atoms_list_path), index=":")
        atoms_list = [atoms_list[i] for i in random_reorder]
        
        node_features = {key: [] for key in species}
        pbar = tqdm(atoms_list, desc="Loading Node Features")
        for atoms in atoms_list:
            chemical_symbols = atoms.get_chemical_symbols()
            node_features_i = atoms.info['node_features']
            for j in range(len(node_features_i)):
                node_features[chemical_symbols[j]].append(node_features_i[j])
            pbar.update(1)
            if np.min([len(value) for key, value in node_features.items()]) > args_dict["min_num_points"]:
                break
        pbar.close()
        node_features = {key: np.array(value) for key, value in node_features.items()}
        
        ## ================== Generate UMAP Plots ==================
        from umap.umap_ import UMAP
        import matplotlib.pyplot as plt
        
        def L2normalize(X):
            return X / np.linalg.norm(X, axis=1)[:, None]
        
        for specie in species:
            umap_model = UMAP(n_components=2, metric="cosine", random_state=0)
            if args_dict["normalize"]:
                node_features_ = L2normalize(node_features[specie])
                soap_features_ = L2normalize(soap_features[specie])
            else:
                node_features_ = node_features[specie]
                soap_features_ = soap_features[specie]
            node_features_umap = umap_model.fit_transform(node_features_)
            node_features_umap = np.array(node_features_umap)
            soap_features_umap = umap_model.fit_transform(soap_features_)
            soap_features_umap = np.array(soap_features_umap)
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(node_features_umap[:, 0], node_features_umap[:, 1], c=energy_per_atom[specie])
            plt.title(f'{model_name} Features for {specie}')
            plt.subplot(1, 2, 2)
            plt.scatter(soap_features_umap[:, 0], soap_features_umap[:, 1], c=energy_per_atom[specie])
            plt.title(f'SOAP Features for {specie}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{specie}_in_{"".join(species)}_{model_name}_block{num_block}_feature_umap.png'))
            
            np.savez(os.path.join(save_dir, f'{specie}_in_{"".join(species)}_{model_name}_block{num_block}_feature_umap.npz'), node_tsne=node_features_umap, soap_tsne=soap_features_umap, energy_per_atom=energy_per_atom[specie])
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMAP Plot for Node Features and SOAP Features")
    parser.add_argument("--species", type=str, nargs="+", default=["Mg", "Si"], help="List of species")
    parser.add_argument("--model_name", type=str, default="jmp-l", help="Model Name")
    parser.add_argument("--r_cut", type=float, default=6.0, help="Cutoff radius for SOAP descriptor")
    parser.add_argument("--n_max", type=int, default=4, help="Maximum number of radial basis functions")
    parser.add_argument("--l_max", type=int, default=4, help="Maximum degree of spherical harmonics")
    parser.add_argument("--min_num_points", type=int, default=20000, help="Minimum number of points")
    parser.add_argument("--normalize", type=bool, default=True, help="Normalize the features")
    args_dict = vars(parser.parse_args())
    main(args_dict)