import inspect
import ray.data
from typing import List, Union
from fmeval.transforms.transform import Transform, Record
from fmeval.util import get_num_actors

NestedTransform = Union[Transform, "TransformPipeline"]


class TransformPipeline:
    def __init__(self, nested_transforms: List[NestedTransform]):
        self.pipeline: List[Transform] = TransformPipeline.flatten(nested_transforms)

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

    def execute(self, dataset: Union[List[Record], ray.data.Dataset]):
        for transform in self.pipeline:
            if isinstance(dataset, ray.data.Dataset):
                """
                Note: if we don't set num_cpus=0, we have to materialize the dataset
                after each call to map, as earlier transforms may take up all of the 
                logical CPUs (by default, actors consume 1 CPU), making it impossible
                to schedule actors for later transforms. We end up with deadlock.
                
                Setting num_cpus=0 resolves the deadlock issue, but this removes the limiter
                on how many actors can be scheduled. It seems that the upper limit provided
                by the concurrency argument isn't respected if num_cpus=0. As a result,
                there can be a *lot* of python processes running. I haven't run into any issues
                with my laptop crashing, and in theory, Ray's AutoScaler should stop creating
                more actors when it detects resource limits have been reached. So perhaps this is
                the best way to go, if we want highest performance.
                
                Materializing after each call to map and using the implicit 1 logical CPU per actor
                seems to be the "safest" option though.
                """
                dataset = dataset.map(
                    transform.__class__,
                    fn_constructor_args=transform.args,
                    fn_constructor_kwargs=transform.kwargs,
                    num_cpus=0,  # actors other than those created by shared_resource shouldn't take up a logical CPU
                    concurrency=(1, get_num_actors()),
                )
            else:
                dataset = list(map(transform, dataset))
        return dataset
