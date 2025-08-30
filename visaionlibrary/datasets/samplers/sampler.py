from typing import Iterator, Optional, Sized
import math
import copy
import numpy as np

from mmengine.registry import DATA_SAMPLERS
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.dataset.sampler import InfiniteSampler

from visaionlibrary.datasets.utils import SampleStatus


@DATA_SAMPLERS.register_module()
class VisaionInfiniteSampler(InfiniteSampler):
    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 bg_ratio: float = 0.1,
                 batch_size: int = 8
                 ) -> None:
        rank, world_size = get_dist_info()  # get rank and world_size
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        # make sure the seed is the same in all processes, so that the order of the samples is the same
        self.seed = sync_random_seed() if seed is None else seed
        
        self.bg_ratio = bg_ratio
        self.batch_size = batch_size
        self.fg_sample_indexes, self.bg_sample_indexes = [], []     # store the indexes of foreground and background samples
        for i, sample_type in enumerate(self.dataset.samples_type):    # noqa, samples_type is a list
            if sample_type == SampleStatus.BACKGROUND:
                self.bg_sample_indexes.append(i)
            elif sample_type == SampleStatus.FOREGROUND:
                self.fg_sample_indexes.append(i)
            else:
                raise ValueError(f"sample_type must be one of {SampleStatus.values()}, but got {sample_type}")
        assert len(self.fg_sample_indexes) > 0, "there must be at least one foreground sample in the dataset"

            
        # obtain the number of foreground and background samples in each batch
        assert 0 <= self.bg_ratio < 1, "bg_ratio must be between 0 and 1"
        self.num_bg_per_batch = math.ceil(self.batch_size * self.bg_ratio) if len(self.bg_sample_indexes) > 0 else 0  # once self.bg_ratio is greater than 0, there must be at least one background sample in each batch
        self.num_fg_per_batch = self.batch_size - self.num_bg_per_batch
        assert self.num_fg_per_batch > 0, "there must be at least one foreground sample in each batch"

        self.size = len(self.dataset)  # the number of samples in the dataset, including foreground and background samples

        self.indices = self._indices_of_rank()

    def _infinite_indices(self) -> Iterator[int]:
        """Infinitely yield a sequence of indices.

        """
        np_random = np.random.RandomState(self.seed)  # make sure the random seed is the same in all processes

        # if the number of self.fg_sample_indexes is less than self.batch_size,
        # we need to repeat the self.fg_sample_indexes to fit at least one batch_size
        fg_sample_indexes = copy.copy(self.fg_sample_indexes)
        if len(fg_sample_indexes) < self.batch_size:
            fg_sample_indexes = fg_sample_indexes * math.ceil(self.batch_size / len(fg_sample_indexes))
            fg_sample_indexes = fg_sample_indexes[:self.batch_size]

        # fill the both fg_sample_indexes and bg_sample_indexes to fit one epoch
        num_batches = math.ceil(len(fg_sample_indexes) / self.num_fg_per_batch)  # the number of batches is computed according to the foreground samples
        # make sure the number of batches can be divided by self.world_size
        if num_batches % self.world_size != 0:
            num_batches = num_batches + self.world_size - num_batches % self.world_size

        total_num_fg_samples = num_batches * self.num_fg_per_batch  # the total number of foreground samples needed
        total_num_bg_samples = num_batches * self.num_bg_per_batch  # the total number of background samples needed
        assert total_num_fg_samples >= len(fg_sample_indexes), "Internal error: total_num_fg_samples should always be >= len(self.fg_sample_indexes)"

        # fill the fg_sample_indexes to fit one epoch
        fg_sample_indexes = fg_sample_indexes + \
                            fg_sample_indexes * math.floor((total_num_fg_samples - len(fg_sample_indexes))/len(fg_sample_indexes)) + \
                            np_random.choice(fg_sample_indexes, size=(total_num_fg_samples - len(fg_sample_indexes))%len(fg_sample_indexes)).tolist()

        # fill the bg_sample_indexes to fit one epoch
        if len(self.bg_sample_indexes) > 0 and self.num_bg_per_batch > 0:
            if total_num_bg_samples < len(self.bg_sample_indexes):
                bg_sample_indexes = np_random.choice(self.bg_sample_indexes, size=total_num_bg_samples).tolist()
            else:
                bg_sample_indexes = copy.copy(self.bg_sample_indexes)
                bg_sample_indexes = bg_sample_indexes + \
                                    bg_sample_indexes * math.floor((total_num_bg_samples - len(bg_sample_indexes))/len(bg_sample_indexes)) + \
                                    np_random.choice(bg_sample_indexes, size=(total_num_bg_samples - len(bg_sample_indexes))%len(bg_sample_indexes)).tolist()
        else:
            bg_sample_indexes = []

        while True:
            if self.shuffle:
                np_random.shuffle(fg_sample_indexes)
                np_random.shuffle(bg_sample_indexes)
            
            # rearrange the fg_sample_indexes and bg_sample_indexes according to batch_size and world_size,
            # so that the batch for each process has the same number of foreground and background samples
            # the number of batches for each process is num_batches // world_size
            all_indexes = []
            num_batches_per_process = num_batches // self.world_size
            for i in range(num_batches_per_process):
                all_indexes += fg_sample_indexes[i*(self.num_fg_per_batch*self.world_size):(i+1)*(self.num_fg_per_batch*self.world_size)] + \
                               bg_sample_indexes[i*(self.num_bg_per_batch*self.world_size):(i+1)*(self.num_bg_per_batch*self.world_size)]
            
            yield from all_indexes

@DATA_SAMPLERS.register_module()
class InfiniteSamplerBatch(InfiniteSampler):
    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 neg_ratio: float = 0.3,
                 neg_epoch_mode: bool = True,
                 batch_size: int = 8
                 ) -> None:
        rank, world_size = get_dist_info()  # 获取rank和world_size
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        # make sure the seed is the same in all processes, so that the order of the samples is the same
        self.seed = sync_random_seed() if seed is None else seed

        self.neg_ratio = neg_ratio
        self.batch_size = batch_size
        self.neg_epoch_mode = neg_epoch_mode
        self.pos_data_indexes, self.neg_data_indexes = [], []     # 存储正负样本索引
        for i, sample_type in enumerate(self.dataset.samples_type):    # noqa, samples_type是list, 在det_dataset.py中定义
            if sample_type == SampleStatus.BACKGROUND:
                self.neg_data_indexes.append(i)
            elif sample_type == SampleStatus.FOREGROUND:
                self.pos_data_indexes.append(i)
            else:
                raise ValueError(f"sample_type must be one of {SampleStatus.values()}, but got {sample_type}")

        # get self.pos_num and self.neg_num
        assert 0 <= self.neg_ratio < 1  # 用户可以指定neg_ratio=0但是不能指定成1
        self.neg_num = int(self.batch_size * self.neg_ratio)
        self.pos_num = self.batch_size - self.neg_num

        # 相当于原来的len(self.dataset)
        self.size = math.ceil(len(self.pos_data_indexes) / self.pos_num)    # 向上取整
        self.indices = self._indices_of_rank()

    def _infinite_indices(self) -> Iterator[int]:
        """Infinitely yield a sequence of indices.
        生成器的内容indices
        [
            [pos_index,...,neg_index,..., len(self.pos_num+self.neg_num)],
               ...
        ]
        """
        np_random = np.random.RandomState(self.seed)  # 加这行的原因是numpy.random在多进程环境下重复生成随机数
        neg_data_indexes = copy.copy(self.neg_data_indexes)
        if self.shuffle:
            np_random.shuffle(neg_data_indexes)
        neg_counter = 0
        while True:
            pos_data_indexes = copy.copy(self.pos_data_indexes)
            if self.shuffle:
                np_random.shuffle(pos_data_indexes)
            # 稀有情况下，如果取完全面的几个pos样本正好剩下一个，而中间特征图又出现1x1这种大小的话，就会导致BN层的计算出现问题
            # 所以，需要把pos_data_indexes进行补全，让他至少能够被self._pos_num整除
            # 这个算的是能够让他被整除的最短长度
            target_length = int(np.ceil((len(pos_data_indexes)) / self.pos_num)) * self.pos_num
            # 这一步是把pos_data_indexes拓展到这个目标长度
            pos_data_indexes = (int(np.ceil(target_length / len(pos_data_indexes))) * pos_data_indexes)[:target_length]

            indices: list = []
            for pos_indexes_i in [pos_data_indexes[i:i + self.pos_num]
                                  for i in range(0, len(pos_data_indexes), self.pos_num)]:
                if len(neg_data_indexes) != 0 and (not self.neg_epoch_mode or len(neg_data_indexes) < self.neg_num):
                    neg_indexes_i = np_random.choice(neg_data_indexes, size=self.neg_num).tolist()
                elif len(neg_data_indexes) != 0 and self.neg_epoch_mode:
                    if neg_counter+self.neg_num > len(neg_data_indexes):
                        tem_neg_data_index = neg_data_indexes[neg_counter:]
                        neg_data_indexes = copy.copy(self.neg_data_indexes)
                        if self.shuffle:
                            np_random.shuffle(neg_data_indexes)
                        neg_data_indexes = tem_neg_data_index + neg_data_indexes
                        neg_counter = 0
                    neg_indexes_i = neg_data_indexes[neg_counter: neg_counter+self.neg_num]
                    neg_counter += self.neg_num
                else:
                    neg_indexes_i = []
                indices.append(pos_indexes_i + neg_indexes_i)

            yield from indices
