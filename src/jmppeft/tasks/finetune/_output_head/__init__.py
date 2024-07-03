from typing import Annotated, TypeAlias

import ll

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
from .graph_direct import (
    GraphBinaryClassificationOutputHead as GraphBinaryClassificationOutputHead,
)
from .graph_direct import (
    GraphBinaryClassificationTargetConfig as GraphBinaryClassificationTargetConfig,
)
from .graph_direct import (
    GraphMulticlassClassificationOutputHead as GraphMulticlassClassificationOutputHead,
)
from .graph_direct import (
    GraphMulticlassClassificationTargetConfig as GraphMulticlassClassificationTargetConfig,
)
from .graph_direct import GraphScalarOutputHead as GraphScalarOutputHead
from .graph_direct import GraphScalarTargetConfig as GraphScalarTargetConfig

GraphTargetConfig: TypeAlias = Annotated[
    GraphScalarTargetConfig
    | GraphBinaryClassificationTargetConfig
    | GraphMulticlassClassificationTargetConfig
    | GradientStressTargetConfig
    | DirectStressTargetConfig,
    ll.Field(discriminator="kind"),
]

NodeTargetConfig: TypeAlias = Annotated[
    NodeVectorTargetConfig | GradientForcesTargetConfig,
    ll.Field(discriminator="kind"),
]
