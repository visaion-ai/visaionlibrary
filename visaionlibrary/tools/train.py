import argparse
import logging
import os
import os.path as osp
import platform
import sys
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

def _extract_page_visaion_keys(data, result=None):
    """
    递归提取以 VISAION 开头的 key 和 value
    """
    if result is None:
        result = {}
    if isinstance(data, dict):
        if 'key' in data and 'value' in data:
            if data['key'].startswith('VISAION'):
                result[data['key']] = data['value']
        for _, value in data.items():
            _extract_page_visaion_keys(value, result)
    elif isinstance(data, list):
        for item in data:
            _extract_page_visaion_keys(item, result)
    return result

def _update_meta_info(meta_info_origin: Union[Dict[str, Any], List[Dict[str, Any]]], task_type: str) -> Dict[str, Any]:
    if task_type == 'det':
        classes = [tmp['name'] for tmp in meta_info_origin]
        palette = [tuple(tmp['color'][:3]) for tmp in meta_info_origin]
        meta_info = {"classes": classes, "palette": palette}
        return meta_info
    elif task_type == 'seg':
        classes = ['background'] + [tmp['name'] for tmp in meta_info_origin]
        palette = [(0, 0, 0)] + [tuple(tmp['color'][:3]) for tmp in meta_info_origin]
        meta_info = {"classes": classes, "palette": palette}
        return meta_info
    elif task_type == 'cls':
        classes = ['background'] + [tmp['name'] for tmp in meta_info_origin]
        palette = [(0, 0, 0)] + [tuple(tmp['color'][:3]) for tmp in meta_info_origin]
        meta_info = {"classes": classes, "palette": palette}
        return meta_info
    elif task_type == 'ins':
        classes = [tmp['name'] for tmp in meta_info_origin]
        palette = [tuple(tmp['color'][:3]) for tmp in meta_info_origin]
        meta_info = {"classes": classes, "palette": palette}
        return meta_info
    else:
        raise ValueError(f"task_type {task_type} not supported")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('template', help='train config file path')
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

def train():
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
    visaion_base_params = _extract_baseParam_visaion_keys(job_args['baseParam'])    # 提取基础参数
    visaion_page_params = _extract_page_visaion_keys(job_args['page'])              # 提取页面参数
    visaion_params = {**visaion_base_params, **visaion_page_params}                 # 合并参数
    # 获取VISAION_DIR
    VISAION_DIR = visaion_params['VISAION_DIR']
    # 更新VISAION_META_INFO
    if 'VISAION_META_INFO' in visaion_params:   
        visaion_params['VISAION_META_INFO'] = _update_meta_info(visaion_params['VISAION_META_INFO'], visaion_params['VISAION_TEMPLATE_TYPE'])
    # 更新VISAION_CHECKPOINT
    if "VISAION_CHECKPOINT" in visaion_params and visaion_params["VISAION_CHECKPOINT"]:  
        visaion_params["VISAION_CHECKPOINT"] = osp.join(VISAION_DIR, 'weights', visaion_params["VISAION_CHECKPOINT"]).replace('\\', '/')
    # 更新VISAION_LOAD_FROM
    if "VISAION_LOAD_FROM" in visaion_params and visaion_params["VISAION_LOAD_FROM"]:  
        visaion_params["VISAION_LOAD_FROM"] = osp.join(VISAION_DIR, 'weights', visaion_params["VISAION_LOAD_FROM"]).replace('\\', '/')
    # 更新VISAION_NUM_CLASSES
    visaion_params['VISAION_NUM_CLASSES'] = len(visaion_params['VISAION_META_INFO']['classes'])
    # 更新VISAION_BACKEND
    if platform.system() == 'Windows':  
        visaion_params['VISAION_BACKEND'] = 'gloo'
    # 更新VISAION_IN_CHANNELS
    if visaion_params["VISAION_COLOR_TYPE"] == "color": 
        visaion_params["VISAION_IN_CHANNELS"] = 3
    elif visaion_params["VISAION_COLOR_TYPE"] == "grayscale":
        visaion_params["VISAION_IN_CHANNELS"] = 1
    elif visaion_params["VISAION_COLOR_TYPE"] == "gray" or visaion_params["VISAION_COLOR_TYPE"] == "grey":
        print_log(f"VISAION_COLOR_TYPE {visaion_params['VISAION_COLOR_TYPE']} is deprecated, use grayscale instead")
        visaion_params["VISAION_IN_CHANNELS"] = 1
        visaion_params["VISAION_COLOR_TYPE"] = 'grayscale'
    else:
        raise ValueError(f"VISAION_COLOR_TYPE {visaion_params['VISAION_COLOR_TYPE']} not supported")
    # 更新VISAION_TRAIN_DATA_ROOT
    visaion_train_data_root = []
    if "VISAION_TRAIN_DATA_ROOT" in visaion_params and visaion_params["VISAION_TRAIN_DATA_ROOT"]:
        for data_name in visaion_params["VISAION_TRAIN_DATA_ROOT"]:
            visaion_train_data_root.append(osp.join(VISAION_DIR, "projects", str(project_id), "datasets", data_name).replace('\\', '/'))
    visaion_params["VISAION_TRAIN_DATA_ROOT"] = "::".join(visaion_train_data_root)
    # 更新VISAION_VAL_DATA_ROOT
    visaion_val_data_root = []
    if "VISAION_VAL_DATA_ROOT" in visaion_params and visaion_params["VISAION_VAL_DATA_ROOT"]:
        for data_name in visaion_params["VISAION_VAL_DATA_ROOT"]:
            visaion_val_data_root.append(osp.join(VISAION_DIR, "projects", str(project_id), "datasets", data_name).replace('\\', '/'))
    visaion_params["VISAION_VAL_DATA_ROOT"] = "::".join(visaion_val_data_root)
    # 更新VISAION_TEST_DATA_ROOT
    visaion_test_data_root = []
    if "VISAION_TEST_DATA_ROOT" in visaion_params and visaion_params["VISAION_TEST_DATA_ROOT"]:
        for data_name in visaion_params["VISAION_TEST_DATA_ROOT"]:
            visaion_test_data_root.append(osp.join(VISAION_DIR, "projects", str(project_id), "datasets", data_name).replace('\\', '/'))
    visaion_params["VISAION_TEST_DATA_ROOT"] = "::".join(visaion_test_data_root)
    # 更新VISAION_TEMPLATE_NAME
    visaion_params['VISAION_TEMPLATE_NAME'] = osp.join(VISAION_DIR, 'templates', visaion_params['VISAION_TEMPLATE_NAME']).replace('\\', '/')
    assert osp.exists(visaion_params['VISAION_TEMPLATE_NAME']), f"VISAION_TEMPLATE_NAME {visaion_params['VISAION_TEMPLATE_NAME']} not exists"
    # 更新VISAION_WORK_DIR
    visaion_params['VISAION_WORK_DIR'] = osp.join(VISAION_DIR, 'projects', str(project_id), job_type, str(job_id)).replace('\\', '/')
    os.makedirs(visaion_params['VISAION_WORK_DIR'], exist_ok=True)
    # 更新VISAION_SCALE_FACTOR
    if "VISAION_SCALE_FACTOR" in visaion_params:
        visaion_params['VISAION_SCALE_FACTOR'] = float(visaion_params['VISAION_SCALE_FACTOR'])
    #  打印 VISAION 参数, 设置环境变量
    print_log('setting environment variables...')
    for k, v in visaion_params.items():
        v = str(v)
        # windows 下使用\\加载config时会失败, 所以需要替换为/
        if platform.system() == 'Windows':
            v = v.replace('\\', '/')
        print_log(f"{k}={v}")
        os.environ[k] = v
    
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config, and update configs
    cfg = Config.fromfile(visaion_params['VISAION_TEMPLATE_NAME'] )
    cfg.work_dir = visaion_params['VISAION_WORK_DIR']
    cfg.launcher = args.launcher
    
    # 打印visaionlibrary版本
    print_log(f"visaionlibrary version: {visaionlibrary.__version__}")
    
    # enable automatic-mixed-precision training
    if "VISAION_AMP" in visaion_params and visaion_params['VISAION_AMP'] is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume is determined in this priority: resume from > auto_resume
    if visaion_params.get('VISAION_CHECKPOINT', None):
        cfg.load_from = visaion_params['VISAION_LOAD_FROM']

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

    # start training
    runner.train()

    runner.logger.info('VISAION_DONE')

if __name__ == '__main__':
    train()
