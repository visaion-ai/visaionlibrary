from typing import Dict
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmengine.registry import METRICS

@METRICS.register_module()
class CocoMetricVisaion(CocoMetric):
    def compute_metrics(self, results: list) -> Dict[str, float]:
        eval_results = super().compute_metrics(results)

        # 新加内容, 澔哥要求只保留mAP, mAP_50
        only_keep_keys = ['bbox_mAP', 'bbox_mAP_50']
        eval_results = {k: v for k, v in eval_results.items() if k in only_keep_keys}
        return eval_results
