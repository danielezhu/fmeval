from abc import abstractmethod
from typing import Any, Dict

Record = Dict[str, Any]


class Transform:
    @abstractmethod
    def __init__(self, **kwargs):
        """"""""

    @abstractmethod
    def __call__(self, record: Record) -> Record:
        """"""
