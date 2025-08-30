import argparse
import os
import json
try:
    import visaionlibrary
except ImportError:
    raise ImportError("visaionlibrary not found, please check your environment")
from visaionlibrary.utils.export_onnx import export_entry

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

def export():
    args = parse_args()

    # 解析参数
    with open(args.template, 'r', encoding='utf-8') as f:
        job_info = json.load(f)

    export_entry(job_info)


if __name__ == '__main__':
    export()
   