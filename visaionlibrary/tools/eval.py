import argparse
import logging
import os
import platform
import os.path as osp
import json
from typing import Union, List, Dict, Any

from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower

try:
    import visaionlibrary
except ImportError:
    raise ImportError("visaionlibrary not found, please check your environment")

def trigger_det_visualization_hook(cfg):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        visualization_hook['draw'] = True
        visualization_hook['test_out_dir'] = "show_dir"
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def trigger_seg_visualization_hook(cfg):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        visualization_hook['draw'] = True
        visualization_hook['interval'] = 1
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def trigger_cls_visualization_hook(cfg, show_dir):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        visualization_hook['out_dir'] = os.path.join(show_dir, "show_dir")
        visualization_hook['interval'] = 1
        visualization_hook['enable'] = True
    return cfg

def _extract_baseParam_visaion_keys(data, result=None):
    """
    递归提取以 VISAION 开头的 key 和 value
    """
    if result is None:
        result = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith('VISAION'):
                result[key] = value
            _extract_baseParam_visaion_keys(value, result)
    elif isinstance(data, list):
        for item in data:
            _extract_baseParam_visaion_keys(item, result)
    return result

def parse_args():
    parser = argparse.ArgumentParser(description='Eval a model')
    parser.add_argument('template', help='eval config file path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def eval():
    args = parse_args()

    # 解析参数
    with open(args.template, 'r', encoding='utf-8') as f:
        job_info = json.load(f)

    # 提取project_id, job_id, job_name, job_args
    project_id = job_info['project_id']     # 项目id
    job_id = job_info['job_id']             # 任务id
    job_type = job_info['job_type']         # 任务名称
    job_args = job_info['job_args']           # 任务参数
    assert project_id is not None, "project_id is not None"
    assert job_id is not None, "job_id is not None"
    assert job_type is not None, "job_type is not None"
    assert job_args is not None, "job_args is not None"

    # 提取以 VISAION 开头的 key 和 value
    visaion_params = _extract_baseParam_visaion_keys(job_args['baseParam'])  # 提取基础参数
    #  打印 VISAION 参数, 设置环境变量
    print_log('setting environment variables...')
    for k, v in visaion_params.items():
        v = str(v)
        # windows 下使用\\加载config时会失败, 所以需要替换为/
        if platform.system() == 'Windows':
            v = v.replace('\\', '/')
        print_log(f"{k}={v}")
        os.environ[k] = v
    
    # 根据参数, 获取配置文件路径, 模型文件路径, 保存eval的文件夹路径, 任务类型
    VISAION_DIR = visaion_params['VISAION_DIR']
    task_type = visaion_params['VISAION_TASK_TYPE']
    config_path = osp.join(VISAION_DIR,visaion_params['VISAION_CONFIG_PATH'])               # 配置文件路径, py文件, 训练后存储的配置文件
    checkpoint_path = osp.join(VISAION_DIR, visaion_params['VISAION_PTH_PATH'])             # 训练后保存的模型文件
    work_dir = osp.join(VISAION_DIR, "projects", str(project_id), "eval", str(job_id)).replace('\\', '/')      # 保存eval的文件夹路径
    data_root = visaion_params['VISAION_TEST_DATA_ROOT']    # 测试数据集路径
    test_data_root = [osp.join(VISAION_DIR, "projects",str(project_id), "datasets", data_i) for data_i in data_root]
    test_data_root = "::".join(test_data_root)
    assert osp.exists(config_path), f'config file {config_path} not found'
    assert osp.exists(checkpoint_path), f'checkpoint file {checkpoint_path} not found'
    assert work_dir is not None, f'work_dir is None'
    assert task_type in ['det', 'seg', 'cls', 'ins'], f'task_type {task_type} not supported'

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(config_path)
    cfg.load_from = checkpoint_path  # 设置模型文件路径
    cfg.work_dir = work_dir  # 设置保存eval的文件夹路径
    cfg.launcher = args.launcher  # 设置任务类型
    cfg.test_dataloader.dataset.data_root = test_data_root  # 设置测试数据集路径
    os.environ["VISAION_WORK_DIR"] = work_dir   # set env variable, used in some hooks, metrics, etc.
    if task_type == 'det':
        trigger_det_visualization_hook(cfg)
    elif task_type == 'seg':
        trigger_seg_visualization_hook(cfg)
    elif task_type == 'cls':
        trigger_cls_visualization_hook(cfg, work_dir)
    elif task_type == 'ins':
        trigger_det_visualization_hook(cfg)
    else:
        raise ValueError(f"task_type {task_type} not supported")

    
    # 打印visaionlibrary版本
    print_log(f"visaionlibrary version: {visaionlibrary.__version__}")
    
    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()

    runner.logger.info("VISAION_DONE")

if __name__ == '__main__':
    eval()
