import argparse
import zipfile
from logging import getLogger
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import requests
import torch
from ll.typecheck import Float, Int
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import override

log = getLogger(__name__)


class _NpzData(TypedDict):
    structure_id: str
    positions: Float[np.ndarray, "n_atoms 3"]
    cell: Float[np.ndarray, "3 3"]
    atomic_numbers: Int[np.ndarray, "n_atoms"]
    energy: Float[np.ndarray, ""]
    stress: Float[np.ndarray, "1 3 3"]
    forces: Float[np.ndarray, "n_atoms 3"]


class MatbenchTrajectoryDataset(Dataset[Data]):
    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    def __init__(
        self,
        base_path: Path,
        energy_linref_path: Path | None = None,
    ) -> None:
        super().__init__()

        # Load the natoms metadata
        self.atoms_metadata = np.load(base_path / "natoms.npy")
        self.data_base_path = base_path / "data"

        self.energy_linref = None
        if energy_linref_path is not None:
            self.energy_linref = np.load(energy_linref_path)

    def __len__(self) -> int:
        return self.atoms_metadata.shape[0]

    def _get_raw_data(self, idx: int) -> _NpzData:
        p = self.data_base_path / f"{idx}.npz"
        assert p.exists(), f"{p} does not exist"
        return cast(_NpzData, np.load(p))

    @override
    def __getitem__(self, idx: int) -> Data:
        npz_data = self._get_raw_data(idx)

        data_dict: dict[str, np.ndarray] = {
            "atomic_numbers": npz_data["atomic_numbers"],
            "cell": npz_data["cell"],
            "pos": npz_data["positions"],
            "energy": np.array(npz_data["energy"]),
            "force": npz_data["forces"],
        }

        # Convert to PT tensors
        data_dict_pt: dict[str, torch.Tensor] = {}
        for k, v in data_dict.items():
            v_pt = torch.from_numpy(v)
            # Check if v is a floating point numpy array
            if v.dtype.kind == "f":
                v_pt = v_pt.float()
            data_dict_pt[k] = v_pt
        data = Data.from_dict(data_dict_pt)

        if self.energy_linref is not None:
            reference_energy = self.energy_linref[data.atomic_numbers].sum()
            data.energy -= reference_energy

        data.y = data.pop("energy")
        data.cell = data.cell.unsqueeze(dim=0)

        return data

    @staticmethod
    def download(args: argparse.Namespace):
        url: str = args.url
        dest: Path = args.dest
        force: bool = args.force or False

        assert url is not None, "Please specify a URL"
        assert dest is not None, "Please specify a destination directory"

        # If the dest exists, check if we need to force download
        if dest.exists():
            if not force:
                log.error(f"{dest} already exists, skipping download")
                return

            log.warning(
                f"{dest} already exists, but force download is enabled"
                " so we will overwrite the existing files."
            )
        else:
            dest.mkdir(parents=True)

        # Download the data
        log.info(f"Downloading data from {url}")

        response = requests.get(url)
        response.raise_for_status()

        # Save the data
        with open(dest / "data.zip", "wb") as f:
            f.write(response.content)

        # Unzip the data
        with zipfile.ZipFile(dest / "data.zip", "r") as zip_ref:
            zip_ref.extractall(dest)

        log.info(f"Data downloaded and extracted to {dest}")

        # Clean up the zip file
        (dest / "data.zip").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    task_subparsers = parser.add_subparsers(dest="task")

    # Add the download task
    download_parser = task_subparsers.add_parser("download")
    download_parser.add_argument(
        "--url",
        type=str,
        help="URL to download the data from",
        default="https://fm-datasets.s3.amazonaws.com/matbench-trajectory-m3gnet.zip",
    )
    download_parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory to save the data",
    )
    download_parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        help="Force download even if the file already exists",
        default=False,
    )
    download_parser.set_defaults(func=MatbenchTrajectoryDataset.download)

    # Parse the arguments and call the appropriate function
    args = parser.parse_args()
    assert args.task is not None, "Please specify a task"

    args.func(args)
