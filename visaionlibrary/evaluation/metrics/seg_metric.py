import os
import os.path as osp
import cv2
import json
import torch
import numpy as np
import networkx as nx
from skimage import morphology
from pathlib import Path
from scipy.ndimage import label as sclabel
from scipy.ndimage import mean as scmean
from collections import OrderedDict
from prettytable import PrettyTable
from typing import Dict, List, Optional, Sequence, Union

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmseg.evaluation import IoUMetric as MMSeg_IoUMetric
from mmengine.registry import METRICS


@METRICS.register_module()
class IoUMetric(MMSeg_IoUMetric):
    """detail ref mmseg.evaluation.IOUMetric
    """

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])[1:]
        total_area_union = sum(results[1])[1:]
        total_area_pred_label = sum(results[2])[1:]
        total_area_label = sum(results[3])[1:]
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes'][1:]

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'IoU':
            #     metrics[key] = val
            # else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # Add classes acc, iou into metrics [add by ldf]
        for key, val in ret_metrics_class.items():
            for i, class_name in enumerate(class_names):
                metrics[key + '_' + class_name] = round(float(val[i]), 2)

        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                # acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                # ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                # acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                # ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics


@METRICS.register_module()
class VisaionMetric(BaseMetric):
    """Visaion evaluation metric. 基于无向图分析，改进多类情况下的计算过程

    """
    def __init__(self,
                 ignore_index: int = 255,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 iou_threshold: float = 0.5,
                 score_threshold: float = 0.5,
                 min_area: int = 20,
                 save_name: str = None,
                 no_argmax=False
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.min_area = min_area
        self.overall_info_array = None
        self.save_name = save_name
        self.no_argmax = no_argmax

    def compute_region_paring(self,
                              pred_logit: torch.tensor,
                              pred_label: torch.tensor,
                              label: torch.tensor
                              ):
        """Calculate Confusion Matrix.
            Args:
                pred_logit (torch.tensor): Prediction origin after activation. The shape is (numClass, H, W).
                pred_label (torch.tensor): Prediction results
                label (torch.tensor): Ground truth segmentation map. The shape is (H, W).
            Returns:
                multiclass_info_list (list)
        """

        pred_activated = pred_logit.max(dim=0, keepdim=True)[0].squeeze()
        pred_activated = pred_activated.cpu().numpy()  # (h, w)
        mask = (label != self.ignore_index)
        pred_label = (pred_label * mask).cpu().numpy()
        label = (label * mask).cpu().numpy()
        num_classes = pred_logit.shape[0]

        # 得到无向图，和对应GT的数量
        graph, gt_nums = self.build_graph(pred_activated, pred_label, label, self.min_area, num_classes)

        multiclass_info_array = np.zeros((num_classes, 100, 3))  # 存储各种类别情况下的信息，一个三维矩阵

        for conf_i in [round(i * 0.01, 2) for i in range(0, 100)]:
            tp_nums, fp_nums = self.analyse_graph(graph, conf_i, self.iou_threshold, self.min_area, num_classes)
            pred_nums = tp_nums + fp_nums  # 统计检出区域的总数
            for subclass in range(num_classes):
                multiclass_info_array[subclass, int(conf_i * 100), 0] = int(gt_nums[subclass])
                multiclass_info_array[subclass, int(conf_i * 100), 1] = int(pred_nums[subclass])
                multiclass_info_array[subclass, int(conf_i * 100), 2] = int(tp_nums[subclass])
        return multiclass_info_array

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:  # data_sample是单个的样本,在一个batch里面
            pred_logit = data_sample['seg_logits']['data']  # c, h, w
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()  # h, w
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)  # h, w
            self.results.append(self.overkill_and_escape(pred_logit, pred_label, label))
            if self.save_name is not None:
                # 只有在指定了保存文件名字的时候才会触发这个多置信度指标计算过程
                # 计算混淆矩阵, by lck
                multiclass_info_array = self.compute_region_paring(pred_logit, pred_label, label)  # num_classes, 100, 3
                if self.overall_info_array is None:
                    self.overall_info_array = multiclass_info_array
                else:
                    self.overall_info_array += multiclass_info_array

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        Added functions:
            1. every classes' metrics into logger
            2. class 'placeholder' clean up.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        if self.save_name is not None:
            # 在之前的基础上额外增加基于overall_info_array(num_classes, 100, 3)的信息处理与保存, gt pred tp
            recall_array = self.overall_info_array[:, :, 2] / self.overall_info_array[:, :, 0]  # [numclass, 100]
            precision_array = self.overall_info_array[:, :, 2] / self.overall_info_array[:, :, 1]  # [numclass, 100]
            f1_score_array = 2 * recall_array * precision_array / (precision_array + recall_array)  # [numclass, 100]

            # 暂时先只存全局的情况，三维矩阵没法存储
            result_array = np.vstack([np.array([round(i / 100., 2) for i in range(100)]),
                                      np.round(f1_score_array[-1], 3),
                                      np.round(recall_array[-1], 3),
                                      np.round(1 - precision_array[-1], 3)])  # 4, 100  第一行是置信度

            def save_xls(result_array, savename="list2Excel.xls"):
                import xlwt
                workbook = xlwt.Workbook()
                sheet = workbook.add_sheet("Sheet")

                for i in range(len(result_array)):
                    for j in range(len(result_array[i])):
                        sheet.write(i, j, result_array[i][j])

                workbook.save(savename)

            save_xls(result_array, self.save_name)

        logger: MMLogger = MMLogger.get_current_instance()

        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        if len(results) != 4:
            return {}

        total_tp_region = sum(results[0])
        total_fp_region = sum(results[1])
        total_gt_region = sum(results[2])
        total_pred_region = sum(results[3])

        ret_metrics = OrderedDict()
        recall_rate = total_tp_region / total_gt_region
        precision_rate = 1 - total_fp_region / total_pred_region
        f_score = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)

        ret_metrics['综合指标'] = f_score
        ret_metrics['区域检出率'] = recall_rate
        ret_metrics['区域过检率'] = 1 - precision_rate

        class_names = self.dataset_meta['classes'][1:]  # 不计算背景
        if not isinstance(self.dataset_meta['classes'], list):
            class_names = list(class_names)
        class_names.append('全局')
        metrics = dict()

        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # Add classes acc, iou into metrics [add by ldf]
        for key, val in ret_metrics_class.items():
            if key == '综合指标':  # 不想返回综合指标这个，这个东西打印显示就行
                continue
            for i, class_name in enumerate(class_names):
                metrics[key + '_' + class_name] = str(round(float(val[i]), 2))

        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            if '区域过检率' in key:  # 如果是precision\Recall，需要显示具体的区域数量
                val = list(map(str, val))  # 把数据都变成list(str)
                # 这个逻辑不想讲，反正结果是对的
                regions = ["(" + i + "/" + j + ")" for i, j in
                           zip(list(map(str, total_fp_region)),
                               list(map(str, total_pred_region)))]
                class_table_data.add_column(key, [i + ' ' + j for i, j in zip(val, regions)])
            elif '区域检出率' in key:
                val = list(map(str, val))  # 把数据都变成list(str)
                # 这个逻辑不想讲，反正结果是对的
                regions = ["(" + i + "/" + j + ")" for i, j in
                           zip(list(map(str, total_tp_region)),
                               list(map(str, total_gt_region)))]
                class_table_data.add_column(key, [i + ' ' + j for i, j in zip(val, regions)])
            else:
                class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def build_graph(pred_activated: np.array,
                    pred_map: np.array,
                    label_map: np.array,
                    min_area: int,
                    num_classes: int) -> (nx.Graph, List, List):
        """根据给定的预测结果，构建无向图，该无向图可以包含类别错误但是面积上重合的目标.
        Args:
            pred_activated (np.array): 预测得到的置信度图
            pred_map (np.array): 预测结果图
            label_map (np.array): 标注图
            min_area (int): 最小面积
            num_classes (int): 类别数量
        Returns: (nx.Graph, List, List)
            graph: 无向图
            gt_nums: 待检测的缺陷数量，是个list，长度等于类别数量,最后一个是不管类别时候的数量
        """
        # 进来的，是个多类的 map 和 label都存在多类的可能性
        # 开始利用检测图、GT图和交集图，构建不同连通域之间关系的无向图，带有类别信息
        graph = nx.Graph()
        label_node_list = []  # GT 的不是双重list是因为不是按照类别分配index而是一起算的，索引的时候只能根据index不然太复杂
        gt_nums = [0] * num_classes  # -1号位准备存不管类别的情况下的区域数量
        pred_node_lists = [[] for _ in range(num_classes - 1)]  # -1 是从根本上不想计算和背景相关的东西
        pred_nums = [0] * num_classes

        # 先把所有的GT放进去，之所以先建立GT而不是和Pred一起建立，是因为可能存在类别不同但是交叠的情况
        label_map_single = morphology.remove_small_objects(label_map != 0, min_size=min_area).astype(np.uint8)
        label_nums, label_compnts, label_stats, _ = cv2.connectedComponentsWithStats(label_map_single, connectivity=8)  # noqa
        gt_nums[-1] = label_nums - 1
        for label_index in range(1, label_nums):
            label_class = (label_map * (label_compnts == label_index)).max()
            node_name = f'g{label_index}_{label_class}_{label_stats[label_index][-1]}'
            graph.add_node(node_name)  # g 结构是 gid_class_area
            label_node_list.append(node_name)  # noqa
            gt_nums[label_class - 1] += 1

        # 因为pred是可能存在完全粘连但是被分开判成两个类别的情况，因此pred_map不能先单类再读取类别这样搞，只能遍历类别这样
        # 同理相交图也是一样，所以也得分类别处理，再往图里添加，同时，建立起来与gt的连接关系
        for class_i in range(1, num_classes):
            # 每一张图中对每一个类别单独计算指标
            pred_map_i = morphology.remove_small_objects(pred_map == class_i, min_size=min_area).astype(np.uint8)

            inter_map_i = (pred_map_i * label_map_single).astype(np.uint8)

            pred_nums_i, pred_compnts_i, pred_stats_i, _ = cv2.connectedComponentsWithStats(pred_map_i)  # noqa
            inter_nums_i, inter_compnts_i, inter_stats_i, _ = cv2.connectedComponentsWithStats(inter_map_i)  # noqa
            pred_nums[class_i - 1] = pred_nums_i - 1
            pred_nums[-1] += pred_nums_i - 1
            for pred_index in range(1, pred_nums_i):  # 从1开始是因为0是背景
                score = (pred_activated * (pred_compnts_i == pred_index)).sum()
                node_name = f'p{pred_index}_{class_i}_{pred_stats_i[pred_index][-1]}_{score}'
                graph.add_node(node_name)  # p 的结构是pid_class_area_置信度总值
                pred_node_lists[class_i - 1].append(node_name)
            for inter_index in range(1, inter_nums_i):
                # 先存储这个inter区域的信息
                node_name = f'i{inter_index}_{class_i}_{inter_stats_i[inter_index][-1]}'
                graph.add_node(node_name)  # i 结构也是 iid_class_area

                # 然后得到这个inter区域对于的pred和gt的index
                inter_region = (inter_compnts_i == inter_index)  # 找出这个index对应的交集区域
                pred_index = (pred_compnts_i * inter_region).max()  # 根据这个交集区域，找出对应pred的index
                label_index = (label_compnts * inter_region).max()  # 以及对应的gt的index

                pred_node = pred_node_lists[class_i - 1][pred_index - 1]  # -1 是因为index是从1开始算的
                label_node = label_node_list[label_index - 1]  # 因为gt的index是全局的，没有类别这一层

                graph.add_edge(node_name, pred_node)  # 由于只会有1个GT和1个pred与这个交集区域相关，所以只需要添加这两个边就行
                graph.add_edge(node_name, label_node)

        # 至此完成图的创建，之所以要和后边的分析过程分开，是因为图建立完之后可能会使用不同的置信度阈值进行分析
        # 不返回pred的信息，因为有可能一个GT有俩Pred相交，或者俩gt有一个pred相交，pred自适应根据gt数量变化 advised from ldf
        return graph, np.array(gt_nums)

    @staticmethod
    def analyse_graph(graph, score_threshold, iou_threshold, min_area, num_classes):
        # 注意，这里的TP区域还是按照标注的区域数量来算的，两个GT如果靠近导致Pred粘连，也是两个一起检出或者不检出
        tp_nums = [0] * num_classes  # 这个长度和num_class一样，是因为-1号准备存全部当单类时候的值
        # 这个就单纯按照pred的区域来算就行
        fp_nums = [0] * num_classes  # 同上

        # 得到图之后，分析每一个连通子图，每个连通子图里边，多个GT和多个pred都联合起来算IOU和置信度，决定这批GT是否检出
        # 这个子图比较恶心，可能有不同类别的Pred和gt相交
        for sub_graph in nx.connected_components(graph):
            # 下面这些都是一起计算检出与否,需要遍历来统计几项内容，
            # GT总面积，GT区域数量，Pred总面积，Pred区域数量，交集总面积，置信度总值，这些都得是list,因为要存不同类别的情况。。。
            gt_area, gt_related_nums, pred_area, pred_related_nums, inter_area, score_sum = \
                [[0] * num_classes for _ in range(6)]
            for node in sub_graph:
                node_type = node[0]  #
                node_info = node.split('_')
                node_class = int(node_info[1])
                node_area = int(node_info[2])
                if node_type == 'i':
                    inter_area[node_class - 1] += node_area
                    inter_area[-1] += node_area  # -1存储的是类别无关的信息
                elif node_type == 'g':
                    gt_area[node_class - 1] += node_area
                    gt_related_nums[node_class - 1] += 1
                    gt_area[-1] += node_area
                    gt_related_nums[-1] += 1
                else:
                    score = float(node_info[-1])
                    pred_area[node_class - 1] += node_area
                    pred_related_nums[node_class - 1] += 1
                    score_sum[node_class - 1] += score
                    pred_area[-1] += node_area
                    pred_related_nums[-1] += 1
                    score_sum[-1] += score
            # 如果这个连通图存在交集区域，那么pred区域按照gt区域数量来算，不然还是实际的pred区域数量
            pred_related_nums = [gt_related_nums[i] if inter_area[i] != 0 else pred_related_nums[i]
                                 for i in range(num_classes)]

            for index in range(num_classes):  # 不想起名叫subclass 因为最后一个位置不是类别的意思
                # 过完这个子图的所有点了，分类分析，就三种情况，1.只有GT 2.只有pred 3.有交集
                if pred_related_nums[index] == 0:
                    continue  # (纯GT的情况这个不用单独处理，只要统计哪些GT检测出来就行)
                # 剩下两种有pred的情况
                # 如果只有Pred没有GT，按照过检来分析
                if gt_related_nums[index] == 0:
                    if pred_area[index] > min_area and score_sum[index] / pred_area[index] > score_threshold:
                        fp_nums[index] += pred_related_nums[index]
                else:  # 这个按照置信度+IOU判定是否检出
                    # 因为有pred有GT必定有inter区域
                    iou = inter_area[index] / (pred_area[index] + gt_area[index] - inter_area[index])
                    if score_sum[index] / pred_area[index] > score_threshold and iou > iou_threshold:
                        tp_nums[index] += gt_related_nums[index]  # 把这部分检出的缺陷数量加到TP里面去
        # 存的是各个类别以及不算类别情况下的信息
        return np.array(tp_nums), np.array(fp_nums)

    def overkill_and_escape(self,
                            pred_logit: torch.tensor,  # (c, h, w)
                            pred_label: torch.tensor,
                            label: torch.tensor,
                            ):
        """Calculate Intersection and Union.
        pred_logit 已经被激活了，不用再处理
        num_classes 包含了背景类别

        返回的，是每个类别的检出区域数量、过检区域数量、GT区域数量和预测区域数量
        """

        pred_activated = pred_logit.max(dim=0, keepdim=True)[0].squeeze()
        pred_activated = pred_activated.cpu().numpy()  # (h, w)
        mask = (label != self.ignore_index)
        pred_label = (pred_label * mask).cpu().numpy()
        label = (label * mask).cpu().numpy()
        num_classes = len(self.dataset_meta['classes'])

        # 得到无向图，和对应的GT还有检测区域的数量
        graph, gt_nums = self.build_graph(pred_activated, pred_label, label, self.min_area, num_classes)

        tp_nums, fp_nums = self.analyse_graph(graph, self.score_threshold, self.iou_threshold, self.min_area, num_classes)

        return tp_nums, fp_nums, gt_nums, tp_nums + fp_nums


@METRICS.register_module()
class VisaionMetricHttp(VisaionMetric):
    """detail ref DLPMetric
    """

    def __init__(self, iou_threshold=0.3, no_argmax=False, output_dir=None, **kwargs) -> None:
        super(VisaionMetricHttp, self).__init__(**kwargs)
        # 设置输出目录
        if os.environ.get("VISAION_WORK_DIR", None):
            self.output_dir = os.environ.get("VISAION_WORK_DIR", None)
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_dir_region_paring = osp.join(self.output_dir, "region_paring")
        os.makedirs(self.output_dir_region_paring, exist_ok=True)
        self.iou_threshold = iou_threshold
        self.no_argmax = no_argmax

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:  # data_sample是单个的样本
            pred_logit = data_sample['seg_logits']['data']  # c, h, w
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()  # h, w
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)  # h, w

            # 获取样本ID
            # sample_id = data_samples[0].get('sample_global_id', 'repeat')
            sample_id = Path(data_sample['img_path']).stem    # 使用图片名作为sample_id
            # sample_id = data_sample['img_id']  # 使用图片id作为sample_id

            # 存储Origin预测结果
            seg_logits_ = pred_logit.detach().cpu().numpy()
            dst_path = os.path.join(self.output_dir, "pred_origin", str(sample_id)+".npy")
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            np.save(dst_path, seg_logits_)

            # 存储Argmax预测结果
            pred_sem_seg = pred_logit.detach().cpu()
            if self.no_argmax:
                _label = label.detach().cpu().numpy().astype(np.uint8)
                _conf = _label * pred_logit.squeeze().detach().cpu().numpy() * 255
            else:
                _label = np.argmax(pred_sem_seg.numpy(), axis=0).astype(np.uint8)
                _conf = (np.max(pred_sem_seg.numpy(), axis=0) * 255).astype(np.uint8)
                _conf = np.where(_label != 0, 1, 0) * _conf

            # 标记前景区域
            labeled_array, num_features = sclabel(_conf > 0)

            # 计算每个区域的均值
            means = scmean(_conf, labeled_array, index=np.arange(1, num_features + 1))

            # 创建一个与输入图像大小相同的输出图像
            output_image = np.zeros_like(_conf)

            # 将均值填充到相应的区域
            for i in range(1, num_features + 1):
                output_image[labeled_array == i] = means[i - 1]

            mask = np.stack([_label, output_image, _conf], axis=0)
            mask = mask.transpose(1, 2, 0)
            dst_path2 = os.path.join(self.output_dir, "pred_origin", str(sample_id) + ".png")
            os.makedirs(os.path.dirname(dst_path2), exist_ok=True)
            cv2.imwrite(dst_path2, mask)

            # 如果未标注的图片 继续下一张
            if torch.all(label == 255):
                continue
            self.results.append(self.overkill_and_escape(pred_logit, pred_label, label))

            # 计算混淆矩阵, by lck
            multiclass_info_list = self.compute_region_paring(pred_logit, pred_label, label)
            self.save_json(multiclass_info_list, osp.join(self.output_dir_region_paring, str(sample_id) + ".json"))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        Added functions:
            1. every classes' metrics into logger
            2. class 'placeholder' clean up.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        if len(results) != 4:
            return {}

        total_tp_region = sum(results[0])
        total_fp_region = sum(results[1])
        total_gt_region = sum(results[2])
        total_pred_region = sum(results[3])

        ret_metrics = OrderedDict()
        recall_rate = total_tp_region / total_gt_region
        precision_rate = 1 - total_fp_region / total_pred_region
        f_score = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)

        ret_metrics['综合指标'] = f_score
        ret_metrics['区域检出率'] = recall_rate
        ret_metrics['区域过检率'] = 1 - precision_rate

        class_names = self.dataset_meta['classes'][1:]  # 不计算背景
        if not isinstance(self.dataset_meta['classes'], list):
            class_names = list(class_names)
        class_names.append('全局')
        metrics = dict()

        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # Add classes acc, iou into metrics [add by ldf]
        for key, val in ret_metrics_class.items():
            if key == '综合指标':  # 不想返回综合指标这个，这个东西打印显示就行
                continue
            for i, class_name in enumerate(class_names):
                metrics[key + '_' + class_name] = str(round(float(val[i]), 2))

        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            if '区域过检率' in key:  # 如果是precision\Recall，需要显示具体的区域数量
                val = list(map(str, val))  # 把数据都变成list(str)
                # 这个逻辑不想讲，反正结果是对的
                regions = ["(" + i + "/" + j + ")" for i, j in
                           zip(list(map(str, total_fp_region)),
                               list(map(str, total_pred_region)))]
                class_table_data.add_column(key, [i + ' ' + j for i, j in zip(val, regions)])
            elif '区域检出率' in key:
                val = list(map(str, val))  # 把数据都变成list(str)
                # 这个逻辑不想讲，反正结果是对的
                regions = ["(" + i + "/" + j + ")" for i, j in
                           zip(list(map(str, total_tp_region)),
                               list(map(str, total_gt_region)))]
                class_table_data.add_column(key, [i + ' ' + j for i, j in zip(val, regions)])
            else:
                class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        self.save_json({
            "iou_threshold": self.iou_threshold,
            "min_area": self.min_area
        }, osp.join(self.output_dir_region_paring, "kwargs.json"))

        return metrics

    def compute_region_paring(self,
                              pred_logit: torch.tensor,
                              pred_label: torch.tensor,
                              label: torch.tensor
                              ):
        """Calculate Confusion Matrix.
            Args:
                pred_logit (torch.tensor): Prediction origin after activation. The shape is (numClass, H, W).
                pred_label (torch.tensor): Prediction results
                label (torch.tensor): Ground truth segmentation map. The shape is (H, W).
            Returns:
                multiclass_info_list (list)
        """

        pred_activated = pred_logit.max(dim=0, keepdim=True)[0].squeeze()
        pred_activated = pred_activated.cpu().numpy()  # (h, w)
        mask = (label != self.ignore_index)
        pred_label = (pred_label * mask).cpu().numpy()
        label = (label * mask).cpu().numpy()
        num_classes = pred_logit.shape[0]

        # 得到无向图，和对应GT的数量
        graph, gt_nums = self.build_graph(pred_activated, pred_label, label, self.min_area, num_classes)

        multiclass_info_list = []  # 存储各种类别情况下的信息 最后一个存的是不分类别情况下的值

        for conf_i in [round(i * 0.01, 2) for i in range(0, 100)]:
            tp_nums, fp_nums = self.analyse_graph(graph, conf_i, self.iou_threshold, self.min_area, num_classes)
            pred_nums = tp_nums + fp_nums  # 统计检出区域的总数
            subclass_info = []
            for subclass in range(num_classes):
                # 不分类的subclass按-1存
                subclass_info.append(
                    {'DefectId': subclass if subclass < num_classes-1 else -1,
                     'gt_nums': int(gt_nums[subclass]),  # 缺陷区域
                     'pred_nums': int(pred_nums[subclass]),  # 全部检测区域
                     'tp_nums': int(tp_nums[subclass])}
                )  # 正确检出区域
            multiclass_info_list.append(subclass_info)
        return multiclass_info_list

    @staticmethod
    def save_json(json_dict: Union[Dict, List] = None, save_path: str = None) -> None:
        """
        保存dict为json文件
        :param json_dict:   字典
        :param save_path:   保存路径
        :return:
        """
        with open(save_path, 'w', encoding="utf-8") as fp:
            json.dump(json_dict, fp, ensure_ascii=False, indent=4)


@METRICS.register_module()
class TransStabilityMetric(BaseMetric):
    def __init__(self,
                 ignore_index: int = 255,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        # data_samples: area_list, prob_list
        if len(data_samples) == 0:
            return  # 负样本
        area_list, prob_list = data_samples
        area_mean, area_std = area_list.mean(), area_list.std()
        prob_mean, prob_std = prob_list.mean(), prob_list.std()

        area_cov = area_std / (area_mean + 1e-8)  # cov:变异系数，也就是波动性的大小
        prob_cov = prob_std / (prob_mean + 1e-8)  # cov:变异系数，也就是波动性的大小

        self.results.append([area_mean, area_std, area_cov, prob_mean, prob_std, prob_cov])

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        res = tuple(zip(*results))
        _, m_area_std, m_area_cov, m_prob_mean, m_prob_std, m_prob_cov = [sum(res[i]) / len(res[i]) for i in range(6)]

        ret_metrics = OrderedDict()
        ret_metrics['平均面积标准差'] = m_area_std
        ret_metrics['平均面积波动百分比'] = m_area_cov * 100.
        ret_metrics['平均置信度均值'] = m_prob_mean
        ret_metrics['平均置信度标准差'] = m_prob_std
        ret_metrics['平均置信度波动百分比'] = m_prob_cov * 100.

        class_table_data = PrettyTable()
        for key, val in ret_metrics.items():
            class_table_data.add_column(key, [str(np.round(val, 3))])
        print_log('\n' + class_table_data.get_string(), logger=logger)
        metrics = dict()
        for key, val in ret_metrics.items():
            metrics[key] = round(val, 3)
        return metrics
