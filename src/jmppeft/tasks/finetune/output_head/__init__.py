from typing import Annotated, TypeAlias

import ll

from ._base import BaseTargetConfig as BaseTargetConfig
from .direct_graph import (
    GraphBinaryClassificationOutputHead as GraphBinaryClassificationOutputHead,
)
from .direct_graph import (
    GraphBinaryClassificationTargetConfig as GraphBinaryClassificationTargetConfig,
)
from .direct_graph import (
    GraphMulticlassClassificationOutputHead as GraphMulticlassClassificationOutputHead,
)
from .direct_graph import (
    GraphMulticlassClassificationTargetConfig as GraphMulticlassClassificationTargetConfig,
)
from .direct_graph import GraphScalarOutputHead as GraphScalarOutputHead
from .direct_graph import GraphScalarTargetConfig as GraphScalarTargetConfig
from .direct_graph_referenced_scalar import (
    ReferencedScalarOutputHead as ReferencedScalarOutputHead,
)
from .direct_graph_referenced_scalar import (
    ReferencedScalarTargetConfig as ReferencedScalarTargetConfig,
)
from .direct_node import NodeVectorOutputHead as NodeVectorOutputHead
from .direct_node import NodeVectorTargetConfig as NodeVectorTargetConfig
from .direct_stress import DirectStressOutputHead as DirectStressOutputHead
from .direct_stress import DirectStressTargetConfig as DirectStressTargetConfig
from .gradient_force_stress import GradientForcesOutputHead as GradientForcesOutputHead
from .gradient_force_stress import (
    GradientForcesTargetConfig as GradientForcesTargetConfig,
)
from .gradient_force_stress import GradientStressOutputHead as GradientStressOutputHead
from .gradient_force_stress import (
    GradientStressTargetConfig as GradientStressTargetConfig,
)

GraphTargetConfig: TypeAlias = Annotated[
    GraphScalarTargetConfig
    | GraphBinaryClassificationTargetConfig
    | GraphMulticlassClassificationTargetConfig
    | GradientStressTargetConfig
    | DirectStressTargetConfig
    | ReferencedScalarTargetConfig,
    ll.Field(discriminator="kind"),
]

NodeTargetConfig: TypeAlias = Annotated[
    NodeVectorTargetConfig | GradientForcesTargetConfig,
    ll.Field(discriminator="kind"),
]
