import copy
from collections import abc
from collections.abc import Callable
from functools import cache, partial
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import wrapt
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import override

from .. import transforms as T
from .dataset_typing import DatasetProtocol, TDataset

log = getLogger(__name__)


def transform(
    dataset: TDataset,
    transform: Callable[[Any], Any],
    copy_data: bool = True,
) -> TDataset:
    """
    Applies a transformation/mapping function to all elements of the dataset.

    Args:
        dataset (Dataset): The dataset to transform.
        transform (Callable): The transformation function.
        copy_data (bool, optional): Whether to copy the data before transforming. Defaults to True.
    """

    class _TransformedDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx):
            nonlocal copy_data, transform

            assert transform is not None, "Transform must be defined."
            data = self.__wrapped__.__getitem__(idx)
            if copy_data:
                data = copy.deepcopy(data)
            data = transform(data)
            return data

    return cast(TDataset, _TransformedDataset(dataset))


def atomref_transform(
    dataset: TDataset,
    refs: dict[str, torch.Tensor],
    keep_raw: bool = False,
) -> TDataset:
    """
    Subtracts the atomrefs from the target properties of the dataset. For a data sample x and atomref property p,
    the transformed property is `x[p] = x[p] - atomref[x.atomic_numbers].sum()`.

    This is primarily used to normalize energies using a "linear referencing" scheme.

    Args:
        dataset (Dataset): The dataset to transform.
        refs (dict[str, torch.Tensor]): The atomrefs to subtract from the target properties.
        keep_raw (bool, optional): Whether to keep the original properties, renamed as `{target}_raw`. Defaults to False.
    """
    # Convert the refs to tensors
    refs_dict: dict[str, torch.Tensor] = {}
    for k, v in refs.items():
        if isinstance(v, list):
            v = torch.tensor(v)
        elif isinstance(v, np.ndarray):
            v = torch.from_numpy(v).float()
        elif not torch.is_tensor(v):
            raise TypeError(f"Invalid type for {k} in atomrefs: {type(v)}")
        refs_dict[k] = v

    return transform(
        dataset,
        partial(T.atomref_transform, refs=refs_dict, keep_raw=keep_raw),
        copy_data=False,
    )


def expand_dataset(dataset: TDataset, n: int) -> TDataset:
    """
    Expands the dataset to have `n` elements by repeating the elements of the dataset as many times as necessary.

    Args:
        dataset (Dataset): The dataset to expand.
        n (int): The desired length of the dataset.
    """
    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"expand_dataset ({n}) must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    og_size = len(dataset)
    if og_size > n:
        raise ValueError(
            f"expand_dataset ({n}) must be greater than or equal to the length of the dataset "
            f"({len(dataset)}) for {dataset.__class__.__qualname__}"
        )

    class _ExpandedDataset(wrapt.ObjectProxy):
        @override
        def __len__(self):
            nonlocal n
            return n

        @override
        def __getitem__(self, index: int):
            nonlocal n, og_size
            if index < 0 or index >= n:
                raise IndexError(
                    f"Index {index} is out of bounds for dataset of size {n}."
                )
            return self.__wrapped__.__getitem__(index % og_size)

        @cache
        def _atoms_metadata_cached(self):
            """
            We want to retrieve the atoms metadata for the expanded dataset.
            This includes repeating the atoms metadata for the elemens that are repeated.
            """

            # the out metadata shape should be (n,)
            nonlocal n, og_size

            metadata = self.__wrapped__.atoms_metadata
            metadata = np.resize(metadata, (n,))
            log.debug(
                f"Expanded the atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    dataset = cast(TDataset, _ExpandedDataset(dataset))
    log.info(f"Expanded dataset {dataset.__class__.__name__} from {og_size} to {n}.")
    return dataset


def first_n_transform(dataset: TDataset, *, n: int) -> TDataset:
    """
    Returns a new dataset that contains the first `n` elements of the original dataset.

    Args:
        dataset (Dataset): The dataset to transform.
        n (int): The number of elements to keep.
    """
    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"first_n ({n}) must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    if len(dataset) < n:
        raise ValueError(
            f"first_n ({n}) must be less than or equal to the length of the dataset "
            f"({len(dataset)}) for {dataset.__class__.__qualname__} "
        )

    class _FirstNDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx: int):
            nonlocal n

            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Index {idx} is out of bounds for dataset of size {n}."
                )

            return self.__wrapped__.__getitem__(idx)

        @override
        def __len__(self):
            nonlocal n
            return n

        @cache
        def _atoms_metadata_cached(self):
            """We only want to retrieve the atoms metadata for the first n elements."""
            nonlocal n

            metadata = self.__wrapped__.atoms_metadata
            og_size = len(metadata)
            metadata = metadata[:n]

            log.info(
                f"Retrieved the first {n} atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    return cast(TDataset, _FirstNDataset(dataset))


def sample_n_transform(dataset: TDataset, *, n: int, seed: int) -> TDataset:
    """
    Similar to first_n_transform, but samples n elements randomly from the dataset.

    Args:
        dataset (Dataset): The dataset to transform.
        n (int): The number of elements to sample.
        seed (int): The random seed to use for sampling.
    """

    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"sample_n ({n}) must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    if len(dataset) < n:
        raise ValueError(
            f"sample_n ({n}) must be less than or equal to the length of the dataset "
            f"({len(dataset)}) for {dataset.__class__.__qualname__} "
        )

    sampled_indices = np.random.default_rng(seed).choice(len(dataset), n, replace=False)

    class _SampleNDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx: int):
            nonlocal n, sampled_indices

            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Index {idx} is out of bounds for dataset of size {n}."
                )

            return self.__wrapped__.__getitem__(sampled_indices[idx])

        @override
        def __len__(self):
            nonlocal n
            return n

        @cache
        def _atoms_metadata_cached(self):
            """We only want to retrieve the atoms metadata for the sampled n elements."""
            nonlocal n, sampled_indices

            metadata = self.__wrapped__.atoms_metadata
            og_size = len(metadata)
            metadata = metadata[sampled_indices]

            log.info(
                f"Retrieved the sampled {n} atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    return cast(TDataset, _SampleNDataset(dataset))


def sample_n_and_get_remaining_transform(
    dataset: TDataset, *, n: int, seed: int
) -> tuple[TDataset, TDataset]:
    """
    Similar to first_n_transform, but it also gets another dataset that contains the remaining elements.

    Args:
        dataset (Dataset): The dataset to transform.
        n (int): The number of elements to sample.
        seed (int): The random seed to use for sampling.
    """

    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"sample_n ({n}) must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    if len(dataset) < n:
        raise ValueError(
            f"sample_n ({n}) must be less than or equal to the length of the dataset "
            f"({len(dataset)}) for {dataset.__class__.__qualname__} "
        )

    sampled_indices = np.random.default_rng(seed).choice(len(dataset), n, replace=False)

    class _SampleNDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx: int):
            nonlocal n, sampled_indices

            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Index {idx} is out of bounds for dataset of size {n}."
                )

            return self.__wrapped__.__getitem__(sampled_indices[idx])

        @override
        def __len__(self):
            nonlocal n
            return n

        @cache
        def _atoms_metadata_cached(self):
            """We only want to retrieve the atoms metadata for the sampled n elements."""
            nonlocal n, sampled_indices

            metadata = self.__wrapped__.atoms_metadata
            og_size = len(metadata)
            metadata = metadata[sampled_indices]

            log.info(
                f"Retrieved the sampled {n} atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    sampled_dataset = cast(TDataset, _SampleNDataset(dataset))

    remaining_indices = np.setdiff1d(np.arange(len(dataset)), sampled_indices)
    remaining_n = len(dataset) - n
    assert remaining_n == len(
        remaining_indices
    ), f"{remaining_n=} != {len(remaining_indices)=}"

    class _RemainingDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx: int):
            nonlocal remaining_n, remaining_indices

            if idx < 0 or idx >= remaining_n:
                raise IndexError(
                    f"Index {idx} is out of bounds for dataset of size {remaining_n}."
                )

            return self.__wrapped__.__getitem__(remaining_indices[idx])

        @override
        def __len__(self):
            nonlocal remaining_n
            return remaining_n

        @cache
        def _atoms_metadata_cached(self):
            """We only want to retrieve the atoms metadata for the sampled remaining_n elements."""
            nonlocal remaining_n, remaining_indices

            metadata = self.__wrapped__.atoms_metadata
            og_size = len(metadata)
            metadata = metadata[remaining_indices]

            log.info(
                f"Retrieved the sampled {remaining_n} atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    remaining_dataset = cast(TDataset, _RemainingDataset(dataset))

    if len(sampled_dataset) + len(remaining_dataset) != len(dataset):
        raise ValueError(
            f"Length of sampled_dataset ({len(sampled_dataset)}) + length of remaining_dataset ({len(remaining_dataset)}) "
            f"must be equal to the length of the dataset ({len(dataset)}) for {dataset.__class__.__qualname__} "
        )

    assert set(sampled_indices).intersection(set(remaining_indices)) == set()

    return sampled_dataset, remaining_dataset


def with_split(
    dataset: TDataset,
    split: torch.Tensor | np.ndarray | list[int] | str | Path,
) -> TDataset:
    """
    Returns a new dataset that contains the elements of the original dataset according to the given split.

    The split should be a 1D array of indices of the dataset, or a file containing such an array.

    Args:
        dataset (Dataset): The dataset to transform.
        split (np.ndarray | list[int] | str | Path): The split to use.
    """

    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"with_split must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    if isinstance(split, (str, Path)):
        split_path = Path(split)
        if not split_path.exists() or not split_path.is_file():
            raise ValueError(f"Split file {split_path} does not exist.")

        loaded_split = np.loadtxt(split_path, dtype=int, comments=None)
    else:
        loaded_split = np.array(split)

    # split must be a 1D array of indices of the dataset
    # therefore, min(split) must be gte 0 and max(split) must be lt len(dataset)
    if min(loaded_split) < 0 or max(loaded_split) >= len(dataset):
        raise IndexError(
            f"Split invalid indices "
            f"({min(loaded_split)=} < 0 or {max(loaded_split)=} >= {len(dataset)=})."
        )

    class _WithSplitDataset(wrapt.ObjectProxy):
        @override
        def __len__(self):
            nonlocal loaded_split
            return len(loaded_split)

        @override
        def __getitem__(self, idx: int):
            nonlocal loaded_split
            if idx < 0 or idx >= len(loaded_split):
                raise IndexError(
                    f"Index {idx} is out of bounds for dataset of size {len(loaded_split)}."
                )

            return self.__wrapped__.__getitem__(loaded_split[idx])

        @cache
        def _atoms_metadata_cached(self):
            """We only want to retrieve the atoms metadata for the computed_split."""
            nonlocal loaded_split

            metadata = self.__wrapped__.atoms_metadata[loaded_split]
            log.info(
                f"Retrieved the split atoms metadata for {self.__class__.__name__} ({len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    return cast(TDataset, _WithSplitDataset(dataset))


def compose(transforms: list[Callable[[TDataset], TDataset]]):
    """
    Composes multiple dataset transforms into a single transform.

    Args:
        transforms (list[Callable[[Dataset], Dataset]]): The transforms to compose.
    """

    def _transform(dataset: TDataset):
        for transform in transforms:
            dataset = transform(dataset)
        return dataset

    return _transform


def with_metadata(
    dataset: Dataset[Data],
    *,
    metadata: str | Path,
):
    """
    Returns a new dataset that contains the elements of the original dataset, but with the atoms metadata
    from the given metadata file.

    The atoms metadata should be a 1D array of the number of atoms in each data sample. This is necessary
    for the "BalancedBatchSampler" to be able to evenly partition batches across training nodes/GPUs
    to maximize GPU memory and compute utilization across nodes.

    Args:
        dataset (Dataset): The dataset to transform.
        metadata (str | Path): The file containing the atoms metadata.
    """

    metadata = Path(metadata)
    if not metadata.exists() or not metadata.is_file():
        raise ValueError(f"Metadata file {metadata} does not exist.")

    metadata_dict = np.load(metadata, allow_pickle=True)
    if "natoms" not in metadata_dict:
        raise ValueError(f"Metadata file {metadata} does not contain 'natoms'.")

    natoms_metadata: np.ndarray = metadata_dict["natoms"]

    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"with_metadata must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    if len(dataset) != len(natoms_metadata):
        raise ValueError(
            f"with_metadata must be used with a dataset that has the same length as the natoms_metadata "
            f"({len(dataset)=} != {len(natoms_metadata)=}) for {dataset.__class__.__qualname__} "
        )

    class _DatasetWithMetadata(wrapt.ObjectProxy):
        @property
        @cache
        def atoms_metadata(self):
            nonlocal natoms_metadata
            return natoms_metadata

    dataset = cast(Dataset[Data], _DatasetWithMetadata(dataset))
    assert isinstance(dataset, DatasetProtocol)
    return dataset
