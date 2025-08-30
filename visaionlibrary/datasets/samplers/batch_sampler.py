from typing import Iterator, List, Union, Iterable
from torch.utils.data import BatchSampler, Sampler
from mmengine.registry import DATA_SAMPLERS

from visaionlibrary.datasets.samplers.sampler import InfiniteSamplerBatch

T_Sampler = Union[
    InfiniteSamplerBatch,
]

@DATA_SAMPLERS.register_module()
class BatchSamplerConstantOne(BatchSampler):
    """
    BatchSampler的子类, 强行将batch_size=1, 因为迭代 sampler(例如InfiniteSamplerBatch) 返回的是一个list(这个list是一个batchsize的数据), 而不是一个int, 且
    所以强行让batch_size=1, 然后在__iter__中, 将list(这个list是一个batchsize的数据)的数据展开成一个batch,
    所以batch_size=1, 然后drop_last=False, 这两个参数废弃
    """
    def __init__(self, sampler: T_Sampler, batch_size: int = 1, drop_last: bool = False) -> None:
        super().__init__(sampler=sampler, batch_size=1, drop_last=False)

    def __iter__(self) -> Iterator[List[int]]:
        batch: List[List[int]] = []
        idx: List[int]
        for idx in self.sampler:    # 迭代sampler, 返回的是一个list(这个list是一个batchsize的数据), 而不是一个int, idx 例如[46, 49, 34, 0, 31, 22, 47, 11, 40, 17, 17, 43], 负样本在后面
            batch.append(idx)
            if len(batch) == self.batch_size:   # batchsize=1, 所以len(batch)=1, 所以len(batch) == self.batch_size, 所以yield batch
                iter_list = [item for sublist in batch for item in sublist]     # batch=[[46, 49, 34, 0, 31, 22, 47, 11, 40, 17, 17, 43]]
                yield iter_list
                batch = []
        # if len(batch) > 0 and not self.drop_last:
        #     yield batch

    def __len__(self) -> int:
        return len(self.sampler)
