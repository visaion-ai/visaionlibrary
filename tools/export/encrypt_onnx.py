from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto import Random
import hashlib
import base64
import json
import argparse

class AESCipher:
    def __init__(self, k: bytes = b'12345678', mode: int = 1):
        self.key = k
        if mode == AES.MODE_ECB:
            self.mode = AES.MODE_ECB
        else:
            raise NotImplementedError

        assert len(k) % 16 == 0
        self.cipher = AES.new(self.key, self.mode)

    def encrypt(self, raw: bytes) -> hex:
        content_padding = pad(raw, AES.block_size)
        cipher_text = self.cipher.encrypt(content_padding)
        return cipher_text.hex()    # 将加密后的二进制数据转换为十六进制字符串格式

    def decrypt(self, encrypted_data: hex) -> bytes:
        # 将十六进制字符串格式的数据转换为二进制格式
        encrypted_data = bytes.fromhex(encrypted_data)
        # 使用解密器进行解密
        decrypted_data = self.cipher.decrypt(encrypted_data)
        # unpad
        decrypted_data = unpad(decrypted_data, AES.block_size)

        return decrypted_data

# python3 /root/2025/projects/visaionsdk/python/encrypt_onnx.py --input /root/2025/projects/visaionsdk/tests_files/seg/S5-3-0903-11-argmax_modify.onnx --output /root/2025/projects/visaionsdk/tests_files/seg/S5-3-0903-11-argmax_modify.visaion
# python3 /root/2025/projects/visaionsdk/python/encrypt_onnx.py --input /root/2025/projects/visaionsdk/tests_files/seg/S5-3-0903-11-argmax_modify_encrypt.onnx --output /root/2025/projects/visaionsdk/tests_files/seg/S5-3-0903-11-argmax_modify_unencrypt.onnx --mode 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input onnx file path')
    parser.add_argument('--output', type=str, help='output dlp file path')
    parser.add_argument('--mode', type=str, default="encrypt", help='AES mode')
    parser.add_argument('--key', type=str, default="12345678", help='AES key')
    args = parser.parse_args()

    cipher = AESCipher(bytes(args.key, encoding='utf8'))
    if args.mode == "encrypt":
         # 加密
        with open(args.input, 'rb') as f:
            plaintext = f.read()
        en_txt = cipher.encrypt(plaintext)  # hex
        # 写文件
        with open(args.output, 'w', encoding='utf8') as f:
            f.write(en_txt)
    else:
        # 读文件
        with open(args.input, 'r', encoding='utf8') as f:
            en_txt_new = f.read()
        # 解密+存储
        with open(args.output, 'wb') as f:
            de_txt = cipher.decrypt(en_txt_new)  # bytes
            f.write(de_txt)