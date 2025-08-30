import re
import numpy as np


def decode_str2mask(mask_str: str, height: int, width: int) -> np.array:
        """
        解压str 到 array, web为了方便传输, 将mask压缩成字符串, 这里解压
        :param mask_str:
        :param height:
        :param width:
        :return:
        """
        _rleDecodeMap = {'Z': '0', 'N': '1'}  # decode 对应map

        if mask_str == "":
            return None

        # groups all encoded pieces together
        mask_list = re.findall(r'(\d+)(\w|\s)', mask_str, re.S) # [('449', 'Z'), ('12', 'N'), ('153', 'Z'), ....]

        # repeat each piece's last char with number
        repeat_list = list(map(lambda x: _rleDecodeMap[x[-1]] * int(x[0]), mask_list))

        # connect && to list
        np_flatten = [int(i) for i in ''.join(repeat_list)]

        # list->mask
        mask = np.reshape(np_flatten, (height, width))
        mask = mask.astype(np.uint8)
        mask = np.ascontiguousarray(mask)
        return mask

def encode_mask_to_str(mask: np.array) -> str:
    """
    将mask编码为字符串
    :param mask: 二值mask数组
    :return: 编码后的字符串
    """
    _rleEncodeMap = {'0': 'Z', '1': 'N'}  # encode 对应map
    
    # 将mask展平为一维数组
    mask_flat = mask.flatten()
    
    # 转换为字符串
    mask_str = ''.join(map(str, mask_flat))
    
    # 使用正则表达式找出连续的相同数字
    pattern = r'(0+|1+)'
    matches = re.finditer(pattern, mask_str)
    
    # 编码结果
    encoded = []
    for match in matches:
        group = match.group()
        count = len(group)
        value = group[0]
        encoded.append(f"{count}{_rleEncodeMap[value]}")
    
    return ''.join(encoded)