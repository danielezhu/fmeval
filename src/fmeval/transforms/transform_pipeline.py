import inspect
import ray.data
from typing import List, Union
from fmeval.transforms.transform import Transform
from fmeval.util import get_num_actors

NestedTransform = Union[Transform, "TransformPipeline"]


class TransformPipeline:
    def __init__(self, nested_transforms: List[NestedTransform]):
        self.pipeline = TransformPipeline.flatten(nested_transforms)

    @staticmethod
    def flatten(nested_transforms: Union[NestedTransform, List[NestedTransform]]) -> List[Transform]:
        if isinstance(nested_transforms, Transform):
            return [nested_transforms]
        if isinstance(nested_transforms, TransformPipeline):
            return nested_transforms.pipeline
        # Can't use iterable unpacking in a list comprehension
        transforms = []
        for nested_transform in nested_transforms:
            transforms.extend(TransformPipeline.flatten(nested_transform))
        return transforms

    def execute(self, dataset: ray.data.Dataset):
        for transform in self.pipeline:
            init_arg_names = inspect.signature(transform.__init__).parameters.keys()
            # We need to materialize the dataset after each transform to ensure that
            # the transformation gets executed in full before the next one starts.
            # Otherwise, we can have deadlock.
            dataset = dataset.map(
                transform.__class__,
                fn_constructor_kwargs={k: v for k, v in transform.__dict__.items() if k in init_arg_names},
                concurrency=(1, get_num_actors()),
            ).materialize()
        return dataset
