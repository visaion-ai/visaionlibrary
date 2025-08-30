import itertools
from typing import Dict, List, Sequence, Union, Optional
import os
import copy
import json
import os.path as osp
from pathlib import Path
from collections import OrderedDict

import torch
from prettytable import PrettyTable
import numpy as np

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine import load
from mmengine.registry import METRICS
from mmpretrain.evaluation.metrics import SingleLabelMetric
from visaionlibrary.datasets.utils import encode_mask_to_str



def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect + 1e-6
    return intersect / union


@METRICS.register_module()
class PrecisionAndRecallsMetric(BaseMetric):
    """ Output Overkill and Miss detections.
    detail ref mmdet.evaluation.CocoMetric
    compute F1 score and
    """


    METRIC_NAMES = ['detect_correct', 'wrong_detect', 'overkill', 'escape', 'detect_correct_rate',
               'escape_rate', 'overkill_rate', 'best_confidence', 'F1_score']
    def __init__(self,
                 iou_threshold=0.3,
                 min_area=64,
                 metric_items: List = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 ) -> None:
        """
           Args:
               confidence_threshold:
               iou_threshold:
               metric_items: ['detect_correct'  # tp
                               'wrong_detect' # mis
                               'overkill'  # fp
                               'escape # fn
                               'detect_correct_rate': 4,  # tp / (tp+mis+fn)
                               'escape_rate': 5,  # fn / (tp+mis+fn)
                               'overkill_rate': 6,  # fp / (tp+fp)
                               'best_confidence': 7,
                               'F1_score': 8
           """
        super().__init__(collect_device=collect_device, prefix=prefix)
        assert iou_threshold >= 0
        self.iou_threshold = iou_threshold
        self.results_extend = {}
        self.min_area = min_area
        self.num_class = None

        if not metric_items:
            metric_items = ['detect_correct_rate', 'escape_rate', 'overkill_rate', 'best_confidence', 'F1_score']
        for metric_item in metric_items:
            assert metric_item in self.METRIC_NAMES, f'{metric_item} must in list {METRICS}'
        self.metric_items = metric_items

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        if self.num_class is None:
            self.num_class = len(self.dataset_meta['classes'])

        # for循环一个batch中的所有图
        for data_sample in data_samples:

            # 预测的所有bbox
            pred_instances = data_sample.get("pred_instances", torch.Tensor([]))
            pred_bboxes_ = pred_instances.get('bboxes', torch.Tensor([])).detach().cpu().numpy()    # [N', 4], N'表示预测的bbox数量
            pred_scores_ = pred_instances['scores'].detach().cpu().numpy()                          # [N']
            pred_labels_ = pred_instances['labels'].detach().cpu().numpy()                          # [N']

            # 真实gt
            gt_bboxes, gt_labels = [], []
            for isinstance in data_sample['instances']:
                gt_bboxes.append(isinstance['bbox'])
                gt_labels.append(isinstance['bbox_label'])

            gt_bboxes = np.array(gt_bboxes, dtype=pred_bboxes_.dtype)   # [N, 4]
            gt_labels = np.array(gt_labels, dtype=pred_labels_.dtype)   # [N]

            # 计算iou
            if len(gt_bboxes) == 0:
                # tp list append 0
                # fp list append len(pred_bbox)
                gt_bboxes = np.array([[0, 0, 0, 0]])
                gt_labels = [None]
            # filter results by confidence threshold [ldf]
            # for each confidence
            for conf_th in range(1, 100):
                score_filter = pred_scores_ >= conf_th * 0.01
                pred_bboxes = pred_bboxes_[score_filter]    # [N'', 4], 过滤掉置信度小于conf_th的bbox
                pred_labels = pred_labels_[score_filter]    # [N'']
                pred_scores = pred_scores_[score_filter]    # [N'']

                if len(pred_bboxes) == 0:
                    pred_bboxes = np.array([[0, 0, 0, 0]])
                    pred_labels = [None]
                    pred_scores = [0]
                ious_argmax = self.match_gt_pred(gt_bboxes, pred_bboxes, pred_scores, gt_labels, pred_labels)

                pred_labels_extend = []
                gt_labels_extend = []
                # label self.num_class is background, -1 is ignore label usually
                for gt_index, gt_i in enumerate(gt_labels): # 遍历gt的label
                    if gt_index in ious_argmax.keys():  # 如果gt index在  gt和pred的匹配中, 取出预测的pred label, 情况1, gt和pred匹配上了, 但不知匹配是否正确
                        pred_labels_extend.append(pred_labels_[ious_argmax[gt_index]])
                        gt_labels_extend.append(gt_i)
                    elif gt_i is not None:  # 如果gt_id 不在gt和pred的匹配中, 将pred的label 置为 一个不存在的类别, 情况2, gt和pred不匹配, 漏检, 有gt没有pred
                        pred_labels_extend.append(self.num_class)
                        gt_labels_extend.append(gt_i)
                for pred_index, pred_i in enumerate(pred_labels): # 遍历pred的label
                    if pred_index not in ious_argmax.values() and pred_i is not None:  # 如果pred index 不在gt和pred的匹配中, 将gt的label 置为 一个不存在的类别, 情况3, pred和gt不匹配, 过检, 有pred没有gt
                        pred_labels_extend.append(pred_i)
                        gt_labels_extend.append(self.num_class)
                if conf_th in self.results_extend.keys():
                    self.results_extend[conf_th].append(
                        {
                            'pred_labels_extend': pred_labels_extend,
                            'gt_labels_extend': gt_labels_extend
                        }
                    )
                else:
                    self.results_extend[conf_th] = [
                        {
                            'pred_labels_extend': pred_labels_extend,
                            'gt_labels_extend': gt_labels_extend
                        }
                    ]

    def match_gt_pred(self, gb, pb, ps, gl, pl) -> dict:    # gt_bboxes, pred_bboxes, pred_scores, gt_labels, pred_labels
        ious = calc_iou(gb, pb)  # gb是gt bbox->[30, 4], pb是pred bbox->[67, 4], eg. (30, 67),表示gt对应pred的iou值
        match_dict = {}
        # filter with iou_threshold
        ious = ious * (ious > self.iou_threshold)  # (n, m), 过滤掉iou小于阈值的bbox
        # filter with label
        ious = ious * (np.tile(np.expand_dims(gl, 1), (1, ious.shape[1])) == np.tile(pl, (ious.shape[0], 1))) # 过滤掉label不匹配的bbox, iou shape->(30, 67)
        index = np.where(ious > 0)  # 返回gt和pred的索引, index[0]是gt的索引,也就是行, index[1]是pred的索引,也就是列, 过滤后index[0].shape->34, index[1].shape->34
        if not len(index[0]):
            return match_dict
        matches = np.array([ious[index], ps[index[1]], index[0], index[1]]).T   # shape->(34, 4), 34表示匹配的bbox对数, 4表示iou, pred score, gt index, pred index
        # order by pred score and iou score
        matches = matches[np.lexsort((-matches[:, 0], -matches[:, 1]))]  # 先按iou降序, 再按pred score降序

        for i, match in enumerate(matches):
            if match[2] not in match_dict.keys():   # key是gt的index, value是pred的index
                match_dict[int(match[2])] = int(match[3])

        return match_dict  # {gt_index: pred_index}, 表示gt的index和pred的index的对应关系

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """

        logger: MMLogger = MMLogger.get_current_instance()
        eval_results = OrderedDict()

        metric_names = {
            'detect_correct': 0,  # tp
            'wrong_detect': 1,  # mis
            'overkill': 2,  # fp
            'escape': 3,  # fn
            'detect_correct_rate': 4,  # tp / (tp+mis+fn)
            'escape_rate': 5,  # fn / (tp+mis+fn)
            'overkill_rate': 6,  # fp / (tp+fp)
            'best_confidence': 7,
            'F1_score': 8
        }

        # compute f1 score
        f1_scores = []
        slm = SingleLabelMetric(average='micro')
        labels = {}
        for conf in range(1, 100):
            # compute most f1
            if conf not in labels.keys():
                labels[conf] = {
                    'y_pred': [],
                    'y_true': []
                }
            for item in self.results_extend[conf]:
                labels[conf]['y_pred'].extend(item['pred_labels_extend'])
                labels[conf]['y_true'].extend(item['gt_labels_extend'])
            precision, recall, f1_score_, support = slm.calculate(np.array(labels[conf]['y_pred']),
                                                                  np.array(labels[conf]['y_true']),
                                                                  average='micro', num_classes=self.num_class+1)
            f1_scores.append(f1_score_)
        f1_score = float(max(f1_scores))
        conf_index = f1_scores.index(f1_score) + 1
        y_pred, y_true = labels[conf_index]['y_pred'], labels[conf_index]['y_true']
        tp = sum([1 if y_pred[i] == y_true[i] else 0 for i in range(len(y_pred))])
        fp = sum([1 if y_pred[i] != y_true[i] and y_true[i] == self.num_class else 0 for i in range(len(y_pred))])
        fn = sum([1 if y_pred[i] != y_true[i] and y_pred[i] == self.num_class else 0 for i in range(len(y_pred))])
        mis = sum([1 if y_pred[i] != y_true[i] and y_pred[i] != self.num_class and y_true[i] != self.num_class else 0
                   for i in range(len(y_pred))])

        result_list = [tp, mis, fp, fn, tp / (tp+mis+fn+1e-6), fn / (tp+mis+fn+1e-6),
                       fp / (tp+fp+1e-6), conf_index*0.01, f1_score]
        headers = ['检测正确', '检错类别', '过检', '漏检',
                   '检出率', '漏检率', '过检率', '最佳置信度', 'f1_score']
        table_data = PrettyTable()

        # filter results
        for i, name in enumerate(metric_names):
            if name in self.metric_items:
                eval_results[name] = result_list[i]
                if i in [4, 5, 6, 7]:
                    table_data.add_column(headers[i], [f'{result_list[i]*100: .2f}%'])
                elif i == 8:
                    table_data.add_column(headers[i], [f'{result_list[i]: .2f}%'])
                else:
                    table_data.add_column(headers[i], [f'{result_list[i]: .2f}'])

        logger.info('\n' + table_data.get_string())

        self.results_extend.clear()
        return eval_results


@METRICS.register_module()
class PrecisionAndRecallsMetricforVisaion(PrecisionAndRecallsMetric):
    def __init__(self,
                 confidence_threshold: float = 0.05,
                 iou_threshold: float = 0.5,
                 output_dir: Optional[str] = None,
                 **kwargs) -> None:

        super().__init__(iou_threshold=iou_threshold)
        self.confidence_threshold = confidence_threshold

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

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        Output:
        - instance_paring
            |- 1.json
            |- 2.json
            ...
        Inside 1.json:
        [
            [{'gtDefectId': None, 'predDefectId': 0, 'count': 3},
            {'gtDefectId': 0, 'predDefectId': None, 'count': 1},
            ],  # for confidence 0.01
            [...]  # for confidence 0.02, ...
        ]

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        super().process(data_batch=data_batch, data_samples=data_samples)

        # for循环一个batch中的所有图
        for data_sample in data_samples:
            sample_id = Path(data_sample['img_path']).stem    # 使用图片名作为sample_id
            # sample_id = data_sample['img_id']  # 使用图片id作为sample_id


            # 预测的所有bbox
            pred_instances = data_sample.get("pred_instances", torch.Tensor([]))
            pred_bboxes = pred_instances.get('bboxes', torch.Tensor([]))
            pred_bboxes = pred_bboxes.detach().cpu().numpy()
            pred_scores = pred_instances['scores'].detach().cpu().numpy()
            pred_labels = pred_instances['labels'].detach().cpu().numpy()
            if "masks" in pred_instances.keys():
                pred_masks = pred_instances['masks'].detach().cpu().numpy()

            # filter results by confidence threshold [ldf]
            score_filter = pred_scores >= self.confidence_threshold
            pred_bboxes = pred_bboxes[score_filter]
            pred_labels = pred_labels[score_filter]
            pred_scores = pred_scores[score_filter]
            if "masks" in pred_instances.keys():
                pred_masks = pred_masks[score_filter]

            # 存储预测结果(原始)
            pred_dict_origin = dict()
            pred_dict_origin['bboxes'] = pred_bboxes
            pred_dict_origin['labels'] = pred_labels
            pred_dict_origin['scores'] = pred_scores
            if "masks" in pred_instances.keys():
                pred_dict_origin['masks'] = pred_masks
            save_dir = osp.join(self.output_dir_pred_origin, str(sample_id) + ".npy")
            np.save(save_dir, pred_dict_origin)

            # 存储预测结果(显示), 用于平台可视化
            pred_dict_show_list = list()
            for i, (bbox, label, score) in enumerate(zip(pred_bboxes, pred_labels, pred_scores)):
                pred_dict_show = dict()
                pred_dict_show['boundingBox'] = self.xyxy2xywh(bbox)
                pred_dict_show['score'] = score.item()
                pred_dict_show['name'] = self.dataset_meta['classes'][label.item()]
                if "masks" in pred_instances.keys():
                    pred_dict_show['mask'] = encode_mask_to_str(pred_masks[i]*1)
                pred_dict_show_list.append(pred_dict_show)
            save_dir = osp.join(self.output_dir_pred_origin, str(sample_id) + ".json")
            self.save_json(pred_dict_show_list, save_dir, same_file_mode='append')

            # 真实gt
            gt_bboxes, gt_labels = [], []
            for isinstance in data_sample['instances']:
                gt_bboxes.append(isinstance['bbox'])
                gt_labels.append(isinstance['bbox_label'])
            gt_bboxes = np.array(gt_bboxes, dtype=pred_bboxes.dtype)
            gt_labels = np.array(gt_labels, dtype=pred_labels.dtype)

            # 计算iou
            if len(gt_bboxes) == 0:
                gt_bboxes = np.array([[0, 0, 0, 0]])
                gt_labels = [None]
            if len(pred_bboxes) == 0:
                pred_bboxes = np.array([[0, 0, 0, 0]])
                pred_labels = [None]
                pred_scores = [0]
            ious_argmax = self.match_gt_pred(gt_bboxes, pred_bboxes, pred_scores, gt_labels, pred_labels)

            # mutli-class
            instance_pairs = []  # gt和pred对
            for gt_index, gt_i in enumerate(gt_labels):  # 这张图上所有gt循环
                if gt_index in ious_argmax.keys() and gt_i == pred_labels[ious_argmax[gt_index]].item():  # gt and pred
                    temp_dict = dict()
                    temp_dict['score'] = pred_scores[ious_argmax[gt_index]]
                    temp_dict['DefectId'] = gt_i
                    temp_dict['gt_nums'] = 1
                    temp_dict['pred_nums'] = 1
                    temp_dict['tp_nums'] = 1
                elif gt_index in ious_argmax.keys() and gt_i != pred_labels[ious_argmax[gt_index]].item():
                    # count 2 instance
                    temp_dict = dict()
                    temp_dict['score'] = pred_scores[ious_argmax[gt_index]]
                    temp_dict['DefectId'] = gt_i
                    temp_dict['gt_nums'] = 1
                    temp_dict['pred_nums'] = 0
                    temp_dict['tp_nums'] = 0
                    instance_pairs.append(temp_dict)
                    temp_dict = dict()
                    temp_dict['score'] = pred_scores[ious_argmax[gt_index]]
                    temp_dict['DefectId'] = pred_labels[ious_argmax[gt_index]].item()
                    temp_dict['gt_nums'] = 0
                    temp_dict['pred_nums'] = 1
                    temp_dict['tp_nums'] = 0
                else:  # gt not pred
                    temp_dict = dict()
                    temp_dict['score'] = None
                    temp_dict['DefectId'] = gt_i
                    temp_dict['gt_nums'] = 1 if gt_i is not None else 0
                    temp_dict['pred_nums'] = 0
                    temp_dict['tp_nums'] = 0
                instance_pairs.append(temp_dict)
            for pred_index, pred_i in enumerate(pred_labels):  # 这张图上所有的pred循环
                if pred_index not in ious_argmax.values():  # pred not gt
                    temp_dict = dict()
                    temp_dict['score'] = pred_scores[pred_index]
                    temp_dict['DefectId'] = pred_i.item() if pred_i is not None else pred_i
                    temp_dict['gt_nums'] = 0
                    temp_dict['pred_nums'] = 1 if pred_i is not None else 0
                    temp_dict['tp_nums'] = 0
                    instance_pairs.append(temp_dict)

            # single-class
            for gt_index, gt_i in enumerate(gt_labels):  # 这张图上所有gt循环
                if gt_index in ious_argmax.keys():  # gt and pred
                    temp_dict = dict()
                    temp_dict['score'] = pred_scores[ious_argmax[gt_index]]
                    temp_dict['DefectId'] = -1 if gt_i is not None else gt_i
                    temp_dict['gt_nums'] = 1 if gt_i is not None else 0
                    temp_dict['pred_nums'] = 1
                    temp_dict['tp_nums'] = 1 if gt_i is not None else 0
                else:  # gt not pred
                    temp_dict = dict()
                    temp_dict['score'] = None
                    temp_dict['DefectId'] = -1 if gt_i is not None else gt_i
                    temp_dict['gt_nums'] = 1 if gt_i is not None else 0
                    temp_dict['pred_nums'] = 0
                    temp_dict['tp_nums'] = 0
                instance_pairs.append(temp_dict)
            for pred_index, pred_i in enumerate(pred_labels):  # 这张图上所有的pred循环
                if pred_index not in ious_argmax.values():  # pred not gt
                    temp_dict = dict()
                    temp_dict['score'] = pred_scores[pred_index]
                    temp_dict['DefectId'] = -1
                    temp_dict['gt_nums'] = 0
                    temp_dict['pred_nums'] = 1
                    temp_dict['tp_nums'] = 0
                    instance_pairs.append(temp_dict)

            # 统计 gt和pred预测对 在不同置信度下的个数
            instance_paring = []
            for i in range(0, 100):  # 0.0~0.99
                conf_i = i * 0.01
                instance_paring_conf_i = dict()     # 置信度conf_i下的预测对
                instance_pairs_tmp = copy.deepcopy(instance_pairs)
                for pair_ in instance_pairs_tmp:
                    score = pair_.pop('score', 0)
                    if score is None:
                        score = 0
                    if score < conf_i:
                        pair_['pred_nums'] = 0
                        pair_['tp_nums'] = 0
                    if pair_['DefectId'] is None:
                        continue
                    if pair_['DefectId'] in instance_paring_conf_i.keys():
                        instance_paring_conf_i[pair_['DefectId']]['gt_nums'] += pair_['gt_nums']
                        instance_paring_conf_i[pair_['DefectId']]['pred_nums'] += pair_['pred_nums']
                        instance_paring_conf_i[pair_['DefectId']]['tp_nums'] += pair_['tp_nums']
                    else:
                        instance_paring_conf_i[pair_['DefectId']] = dict()
                        instance_paring_conf_i[pair_['DefectId']]['gt_nums'] = pair_['gt_nums']
                        instance_paring_conf_i[pair_['DefectId']]['pred_nums'] = pair_['pred_nums']
                        instance_paring_conf_i[pair_['DefectId']]['tp_nums'] = pair_['tp_nums']

                instance_paring_conf_i_result = []
                for _id in instance_paring_conf_i.keys():
                    tmp_dict_ = {
                        'DefectId': int(_id),
                        'gt_nums': instance_paring_conf_i[_id]['gt_nums'],
                        'pred_nums': instance_paring_conf_i[_id]['pred_nums'],
                        'tp_nums': instance_paring_conf_i[_id]['tp_nums']
                    }
                    instance_paring_conf_i_result.append(tmp_dict_)
                instance_paring.append(instance_paring_conf_i_result)
            # actrually used
            save_dir = osp.join(self.output_dir_instance_paring, str(sample_id) + ".json")
            self.save_json(instance_paring, save_dir, same_file_mode='add')

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
