import onnxruntime as ort
import numpy as np
import json
import cv2
import argparse


def preprocess(image, input_shape):
    # 等比例缩放
    h, w = image.shape[:2]
    scale = min(input_shape[2] / h, input_shape[3] / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    image = cv2.resize(image, (new_w, new_h))

    # 填充
    image = cv2.copyMakeBorder(image, 0, input_shape[2] - new_h, 0, input_shape[3] - new_w, cv2.BORDER_CONSTANT, value=[114, 114, 114])
    
    image = image.transpose(2, 0, 1)  # H, W, C -> C, H, W
    image = np.array([image for _ in range(input_shape[0])], dtype=np.float32)  # C, H, W -> B, C, H, W
    image = np.ascontiguousarray(image, dtype=np.float32)

    return image, scale, input_shape[2]-new_h, input_shape[3]-new_w

def postprocess(bboxes, image_shape_hw, input_shape_hw, pad_bottom, pad_right, threshold=0.5, nms_threshold=0.5):
    # 仅选取第1张图
    bboxes_first = bboxes[0]
    # xyxy转换成xywh
    bboxes_first[:, 2] = bboxes_first[:, 2] - bboxes_first[:, 0]
    bboxes_first[:, 3] = bboxes_first[:, 3] - bboxes_first[:, 1]
    # 使用nms对结果进行筛选
    bboxes_first_filter_index = cv2.dnn.NMSBoxes(bboxes_first[:, 0:4], bboxes_first[:, 4], threshold, nms_threshold)
    bboxes_first_filter = bboxes_first[bboxes_first_filter_index]
    bboxes_first_bboxes = bboxes_first_filter[:, 0:4]
    # 获取类别
    bboxes_first_classes = bboxes_first_filter[:, 5:].argmax(axis=-1)
    # 获取分数
    bboxes_first_scores = bboxes_first_filter[:, 4]
    
    # bboxes还原到原始图像尺寸
    orign_h, orign_w = image_shape_hw
    model_h, model_w = input_shape_hw
    image_h, image_w = model_h - pad_bottom, model_w - pad_right
    bboxes_first_bboxes[:, 0] = bboxes_first_bboxes[:, 0] * 1.0 * orign_w / image_w
    bboxes_first_bboxes[:, 1] = bboxes_first_bboxes[:, 1] * 1.0 * orign_h / image_h
    bboxes_first_bboxes[:, 2] = bboxes_first_bboxes[:, 2] * 1.0 * orign_w / image_w
    bboxes_first_bboxes[:, 3] = bboxes_first_bboxes[:, 3] * 1.0 * orign_h / image_h
    # xyhw->xyxy->clip->xyhw
    bboxes_first_bboxes_x1 = bboxes_first_bboxes[:, 0]
    bboxes_first_bboxes_y1 = bboxes_first_bboxes[:, 1]
    bboxes_first_bboxes_x2 = bboxes_first_bboxes_x1 + bboxes_first_bboxes[:, 2]
    bboxes_first_bboxes_y2 = bboxes_first_bboxes_y1 + bboxes_first_bboxes[:, 3]
    bboxes_first_bboxes[:, 0] = np.clip(bboxes_first_bboxes_x1, 0, orign_w)
    bboxes_first_bboxes[:, 1] = np.clip(bboxes_first_bboxes_y1, 0, orign_h)
    bboxes_first_bboxes[:, 2] = np.clip(bboxes_first_bboxes_x2, 0, orign_w)
    bboxes_first_bboxes[:, 3] = np.clip(bboxes_first_bboxes_y2, 0, orign_h)
    bboxes_first_bboxes[:, 2] = bboxes_first_bboxes[:, 2] - bboxes_first_bboxes[:, 0]
    bboxes_first_bboxes[:, 3] = bboxes_first_bboxes[:, 3] - bboxes_first_bboxes[:, 1]


    return bboxes_first_bboxes, bboxes_first_classes, bboxes_first_scores

def infer_onnx(model_path: str, image_path: str, output_path: str):
    # 加载模型
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name   # 获取模型输入, 输出
    input_shape = session.get_inputs()[0].shape   # 获取模型输入形状
    metainfo = session.get_modelmeta().custom_metadata_map['metainfo'] 
    classes_dict = json.loads(metainfo).get('classes')


    # 读取图片
    image = cv2.imread(image_path)
    input_image, scale, pad_bottom, pad_right = preprocess(image, input_shape)


    # 推理
    result = session.run(None, {input_name: input_image.astype(np.float32)})[0]

    # 后处理
    bboxes, classes, scores = postprocess(result, image.shape[:2], input_shape[2:], pad_bottom, pad_right, threshold=0.5, nms_threshold=0.5)

    # 绘制图片
    for bbox, class_id, score in zip(bboxes, classes, scores):
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f"{classes_dict[str(classes[class_id])]} {score:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    cv2.imwrite(output_path, image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='input onnx file path')
    parser.add_argument('--image', type=str, help='input image file path')
    parser.add_argument('--output', type=str, help='output image file path')
    args = parser.parse_args()

    infer_onnx(args.model, args.image, args.output)