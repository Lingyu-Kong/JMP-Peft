from ase import Atoms
from ase.io import read
from dscribe.descriptors import SOAP
from typing import List, Literal
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
import os


def main(args_dict: dict):
    tsne_type:Literal["cosine", "euclidean", "pca_normalized"] = args_dict["tsne_type"]
    species = args_dict["species"]
    model_name:Literal["jmp-s", "jmp-l", "jmp-s-finetuned"] = args_dict["model_name"]
    
    save_dir = "./{}-{}-tsne-{}".format("".join(species), model_name, tsne_type)
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
    
    ## ================== Load ML Node Features ==================
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
        for i, atoms in enumerate(atoms_list):
            chemical_symbols = atoms.get_chemical_symbols()
            node_features_i = atoms.info['node_features']
            for j in range(len(node_features_i)):
                node_features[chemical_symbols[j]].append(node_features_i[j])
            pbar.update(1)
            if np.min([len(value) for key, value in node_features.items()]) > args_dict["min_num_points"]:
                break
        pbar.close()
        node_features = {key: np.array(value) for key, value in node_features.items()}
        for specie in species:
            print(f"Node features for {specie}: {node_features[specie].shape}")
            print(f"SOAP features for {specie}: {soap_features[specie].shape}")
            print(f"Energy per atom for {specie}: {energy_per_atom[specie].shape}")
        
    ## ================== Calculate and Plot TSNE ==================
        for specie in species:
            if tsne_type == "pca_normalized":
                pca = PCA(n_components=50)
                node_features_i = node_features[specie]
                soap_features_i = soap_features[specie]
                node_features_pca = pca.fit_transform(node_features_i)
                soap_features_pca = pca.fit_transform(soap_features_i)
                
                tsne = TSNE(n_components=2, random_state=0)
                node_features_tsne = tsne.fit_transform(node_features_pca)
                soap_features_tsne = tsne.fit_transform(soap_features_pca)
            elif tsne_type == "cosine":
                def cosine_distance(X):
                    X_normalized = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
                    X_distance_matrix = 1 - np.dot(X_normalized, X_normalized.T)
                    X_distance_matrix = np.clip(X_distance_matrix, 0, 2)
                    return X_distance_matrix
                node_features_i = node_features[specie]
                node_features_distance_matrix = cosine_distance(node_features_i)
                soap_features_i = soap_features[specie]
                soap_features_distance_matrix = cosine_distance(soap_features_i)
                node_features_tsne = TSNE(n_components=2, random_state=0, init="random", metric="precomputed").fit_transform(node_features_distance_matrix)
                soap_features_tsne = TSNE(n_components=2, random_state=0, init="random", metric="precomputed").fit_transform(soap_features_distance_matrix)
            elif tsne_type == "euclidean":
                node_features_tsne = TSNE(n_components=2, random_state=0).fit_transform(node_features[specie])
                soap_features_tsne = TSNE(n_components=2, random_state=0).fit_transform(soap_features[specie])
            else:
                raise ValueError(f"Invalid TSNE Type: {tsne_type}")

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            sc1 = plt.scatter(node_features_tsne[:, 0], node_features_tsne[:, 1], c=energy_per_atom[specie])
            plt.title(f'{model_name} Features for {specie}')
            plt.subplot(1, 2, 2)
            sc2 = plt.scatter(soap_features_tsne[:, 0], soap_features_tsne[:, 1], c=energy_per_atom[specie], cmap='viridis')
            plt.title(f'SOAP Features for {specie}')
            plt.tight_layout()
            plt.colorbar(sc2, label='Energy per Atom')  # color bar
            plt.savefig(os.path.join(save_dir, f'{specie}_in_{"".join(species)}_{model_name}_block{num_block}_feature_tsne_{tsne_type}.png'))
            
            np.savez(os.path.join(save_dir, f'{specie}_in_{"".join(species)}_{model_name}_block{num_block}_feature_tsne_{tsne_type}.npz'), node_tsne=node_features_tsne, soap_tsne=soap_features_tsne, energy_per_atom=energy_per_atom[specie])
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsne_type", type=str, default="cosine", help="Type of TSNE to use")
    parser.add_argument("--species", type=str, nargs="+", default=["Mg","Si"], help="Species to consider")
    parser.add_argument("--model_name", type=str, default="jmp-l", help="Model name")
    parser.add_argument("--r_cut", type=float, default=10.0, help="Cutoff radius for SOAP")
    parser.add_argument("--n_max", type=int, default=6, help="Max number of radial functions")
    parser.add_argument("--l_max", type=int, default=6, help="Max number of angular functions")
    parser.add_argument("--min_num_points", type=int, default=20000, help="Minimum number of points to consider")
    args_dict = vars(parser.parse_args())
    main(args_dict)
