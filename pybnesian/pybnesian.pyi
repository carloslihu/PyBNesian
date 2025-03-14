from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

class BayesianNetworkType:
    pass

class GaussianNetworkType(BayesianNetworkType):
    pass

class SemiparametricBNType(BayesianNetworkType):
    pass

class DiscreteBNType(BayesianNetworkType):
    pass

class BayesianNetwork:
    def __init__(self, nodes: Optional[List[str]] = None) -> None: ...
    def nodes(self) -> List[str]: ...
    def arcs(self) -> List[Tuple[str, str]]: ...
    def node_types(self) -> Dict[str, str]: ...
    def fit(self, data: pd.DataFrame) -> None: ...
    def logl(self, data: pd.DataFrame) -> np.ndarray: ...
    def slogl(self, data: pd.DataFrame) -> float: ...
    def predict(self, data: pd.DataFrame) -> pd.DataFrame: ...
    def save(self, path: str) -> None: ...
    def sample(self, size: int, seed: int) -> pd.DataFrame: ...
    @classmethod
    def load(cls, path: str) -> "BayesianNetwork": ...

class SemiparametricBN(BayesianNetwork):
    pass

class GaussianNetwork(BayesianNetwork):
    pass

class DiscreteBN(BayesianNetwork):
    pass

# Add other classes and methods as needed
