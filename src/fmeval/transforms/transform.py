from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
from ray.data import Dataset

Record = Dict[str, Any]


class Transform(ABC):
    def __init__(self, input_keys: List[str], output_keys: List[str], *args, **kwargs):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.args = (input_keys, output_keys) + args
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, record: Record) -> Record:
        """"""


class Aggregation(ABC):

    def __init__(self, aggregation_columns: List[str]):
        self.aggregation_columns = aggregation_columns

    @abstractmethod
    def __call__(self, ds: Dataset) -> Record:
        """"""
