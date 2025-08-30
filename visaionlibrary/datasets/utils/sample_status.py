from enum import Enum


class SampleStatus(Enum):
    """
    1未标注 2背景 3已标注
    """
    RAW = 1
    BACKGROUND = 2
    FOREGROUND = 3
