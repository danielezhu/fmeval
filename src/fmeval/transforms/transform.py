from abc import ABC, abstractmethod
from typing import Any, Dict, List
from ray.data import Dataset

Record = Dict[str, Any]


class Transform(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        """"""""

    @abstractmethod
    def __call__(self, record: Record) -> Record:
        """"""

    @property
    @abstractmethod
    def output_keys(self):
        pass


class Aggregation(ABC):

    def __init__(self, aggregation_columns: List[str]):
        self.aggregation_columns = aggregation_columns

    @abstractmethod
    def __call__(self, ds: Dataset) -> Record:
        """"""
