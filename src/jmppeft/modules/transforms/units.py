from typing import Literal

from torch_geometric.data.data import BaseData

VALID_UNITS = ("eV", "kcal/mol", "hartree", "bohr", "angstrom")
Unit = Literal["eV", "kcal/mol", "hartree", "bohr", "angstrom"]


def _determine_factor(from_: Unit, to: Unit):
    if from_ == to:
        return 1.0

    match (from_, to):
        case ("eV", "kcal/mol"):
            return 23.061
        case ("eV", "hartree"):
            return 0.0367493
        case ("kcal/mol", "eV"):
            return 1 / 23.061
        case ("kcal/mol", "hartree"):
            return 1 / 627.509
        case ("hartree", "eV"):
            return 1 / 0.0367493
        case ("hartree", "kcal/mol"):
            return 627.509
        case ("bohr", "angstrom"):
            return 0.529177
        case ("angstrom", "bohr"):
            return 1 / 0.529177
        case _:
            raise ValueError(f"Cannot convert {from_} to {to}")


def update_units(from_: Unit, to: Unit, *, attributes: list[str] = ["y", "force"]):
    factor = _determine_factor(from_, to)

    def _update_units(data: BaseData):
        nonlocal factor

        for attr in attributes:
            if (value := getattr(data, attr, None)) is None:
                continue
            setattr(data, attr, value * factor)

        return data

    return _update_units


def update_pyg_data_units(
    data: BaseData,
    attributes: list[str],
    *,
    from_: Unit,
    to: Unit,
):
    factor = _determine_factor(from_, to)
    for attr in attributes:
        if (value := getattr(data, attr, None)) is None:
            continue
        setattr(data, attr, value * factor)

    return data
