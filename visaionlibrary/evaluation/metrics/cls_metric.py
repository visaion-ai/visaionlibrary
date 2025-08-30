from typing import Dict, List, Sequence, Union, Optional
import os
import os.path as osp
import copy

import json
from pathlib import Path
import numpy as np
import torch
from mmengine import load
from mmengine.registry import METRICS
from mmpretrain.evaluation.metrics import Accuracy


@METRICS.register_module()
class AccuracyVisaion(Accuracy):
    def compute_metrics(self, results: List):
        metrics = super().compute_metrics(results)

        # 计算混淆矩阵
        target = torch.cat([res['gt_label'] for res in results])
        pred_ = torch.max(torch.stack([res['pred_score'] for res in results]), dim=1)[1].cpu().numpy()
        target_ = target.cpu().numpy()
        num_classes = np.max([pred_, target_]) + 1
        cm = [[0 for j in range(num_classes)] for i in range(num_classes)]
        self.confusion_matrix(pred_, target_, cm)
        cm_dict = dict()
        for i, target_i in enumerate(cm):
            cm_dict[i] = dict()
            for j, pred_i in enumerate(target_i):
                cm_dict[i][j] = cm[i][j]
        metrics['confusion_matrix'] = cm_dict
        return metrics

    def confusion_matrix(self, preds, labels, conf_matrix):
        for p, t in zip(preds, labels):
            conf_matrix[t][p] += 1
        return conf_matrix

@METRICS.register_module()
class AccuracyVisaionHttp(AccuracyVisaion):
    def __init__(self,
                 topk: Union[int, Sequence[int]] = (1, ),
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 ) -> None:
        super().__init__(topk=topk, thrs=thrs, collect_device=collect_device, prefix=prefix)

        # 设置输出目录
        if os.environ.get("VISAION_WORK_DIR", None):
            # 设置输出目录
            self.output_dir = os.environ.get("VISAION_WORK_DIR", None)
        else:
            self.output_dir = output_dir
        assert self.output_dir is not None
        os.makedirs(self.output_dir, exist_ok=True)

        # pred_origin
        self.output_dir_pred_origin = osp.join(self.output_dir, "pred_origin")
        os.makedirs(self.output_dir_pred_origin, exist_ok=True)

        # instance_paring
        self.output_dir_instance_paring = osp.join(self.output_dir, "instance_paring")
        os.makedirs(self.output_dir_instance_paring, exist_ok=True)


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """

        super().process(data_batch=data_batch, data_samples=data_samples)

        for data_sample in data_samples:
            sample_id = Path(data_sample['img_path']).stem    # 使用图片名作为sample_id

            # 预测结果
            pred_label = data_sample['pred_label'].item()
            pred_name = self.dataset_meta['classes'][pred_label]
            pred_score = data_sample['pred_score'][pred_label].item()

            # 真实结果
            gt_label = data_sample['gt_label'].item()

            # 存储预测结果, 用于平台显示
            pred_dict_show_list = list()
            pred_dict_show = dict()
            pred_dict_show['pred_label'] = pred_label
            pred_dict_show['pred_name'] = pred_name
            pred_dict_show['pred_score'] = pred_score
            pred_dict_show['gt_label'] = gt_label
            pred_dict_show_list.append(pred_dict_show)
            save_dir = osp.join(self.output_dir_pred_origin, str(sample_id) + ".json")
            self.save_json(pred_dict_show_list, save_dir, same_file_mode='append')

                
    @staticmethod
    def save_json(json_dict: Union[Dict, List] = None, save_path: str = None, same_file_mode: str = None) -> None:
        """
        保存dict为json文件
        :param json_dict:
        :param save_path:   保存路径
        :same_file_mode: default is update
        :return:
        """

        # 如果文件里面已经有东西了，拿出来追加
        if osp.exists(save_path):
            ori_info = load(save_path)  # [{}, {}]
            # for each ins add
            if same_file_mode == 'add':
                for i, ins in enumerate(json_dict):
                    if isinstance(ins, List):
                        json_dict[i].extend(ori_info[i])
            elif same_file_mode == 'append':
                json_dict.extend(ori_info)
        with open(save_path, 'w', encoding="utf-8") as fp:
            json.dump(json_dict, fp, ensure_ascii=False, indent=4)
