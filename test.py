# -------------------------------------#
#       创建YOLO类
# -------------------------------------#
import colorsys
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.efficientdet import EfficientDetBackbone
from utils.utils import non_max_suppression, bbox_iou, decodebox, letterbox_image, efficientdet_correct_boxes

image_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]


def preprocess_input(image):
    image /= 255
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)
    image -= mean
    image /= std
    return image


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和phi都需要修改！
# --------------------------------------------#
class EfficientDet(object):
    _defaults = {
        "model_path": 'model_data/efficientdet-d0.pth',
        "classes_path": 'model_data/coco_classes.txt',
        "phi": 0,
        "confidence": 0.4,
        "iou": 0.5,
        "cuda": False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Efficientdet
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.net = EfficientDetBackbone(len(self.class_names), self.phi).eval()

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        if self.cuda:
            state_dict = torch.load(self.model_path)
        else:
            state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(state_dict)
        self.net = nn.DataParallel(self.net)
        if self.cuda:
            self.net = self.net.cuda()
        print('Finished!')

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (image_sizes[self.phi], image_sizes[self.phi])))
        photo = np.array(crop_img, dtype=np.float32)
        photo = np.transpose(preprocess_input(photo), (2, 0, 1))
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            _, regression, classification, anchors = self.net(images)

            # regression = decodebox(regression, anchors, images)
            # detection = torch.cat([regression, classification], axis=-1)
            # batch_detections = non_max_suppression(detection, len(self.class_names),
            #                                        conf_thres=self.confidence,
            #                                        nms_thres=self.iou)
            return classification


def show_CAM(image_path, feature_maps, class_id, all_ids=90, show_one_layer=True):
    """
    feature_maps: this is a list [tensor,tensor,tensor], tensor shape is [1, 3, N, N, all_ids]
    """
    SHOW_NAME = ["class"]
    img_ori = cv2.imread(image_path)
    layers0 = feature_maps[0].reshape([-1, all_ids])
    layers1 = feature_maps[1].reshape([-1, all_ids])
    layers2 = feature_maps[2].reshape([-1, all_ids])
    layers3 = feature_maps[3].reshape([-1, all_ids])
    layers4 = feature_maps[4].reshape([-1, all_ids])
    layers = torch.cat([layers0, layers1, layers2, layers3, layers4], 0)
    # score_max_v = layers[:, 4].max()  # compute max of score from all anchor
    # score_min_v = layers[:, 4].min()  # compute min of score from all anchor
    class_max_v = layers[:, class_id].max()  # compute max of class from all anchor
    class_min_v = layers[:, class_id].min()  # compute min of class from all anchor
    all_ret = [[]]
    for j in range(5):  # layers
        layer_one = feature_maps[j]
        # compute max of score from three anchor of the layer
        # anchors_score_max = layer_one[0, ..., 4].max(0)[0]
        # compute max of class from three anchor of the layer
        anchors_class_max = layer_one[0, ..., 5 + class_id].max(0)[0]

        # scores = ((anchors_score_max - score_min_v) / (
        #         score_max_v - score_min_v))

        classes = ((anchors_class_max - class_min_v) / (
                class_max_v - class_min_v))

        layer_one_list = []
        # layer_one_list.append(scores)
        layer_one_list.append(classes)
        # layer_one_list.append(scores * classes)
        for idx, one in enumerate(layer_one_list):
            layer_one = one.cpu().numpy()
            ret = ((layer_one - layer_one.min()) / (layer_one.max() - layer_one.min())) * 255
            ret = ret.astype(np.uint8)
            gray = ret[:, :, None]
            ret = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            if not show_one_layer:
                all_ret[j].append(cv2.resize(ret, (img_ori.shape[1], img_ori.shape[0])).copy())
            else:
                ret = cv2.resize(ret, (img_ori.shape[1], img_ori.shape[0]))
                show = ret * 0.8 + img_ori * 0.2
                show = show.astype(np.uint8)
                cv2.imshow(f"one_{SHOW_NAME[idx]}", show)
                cv2.imwrite('./cam_results/head' + str(j) + 'layer' + str(idx) + SHOW_NAME[idx] + ".jpg", show)
                # cv2.imshow(f"map_{SHOW_NAME[idx]}", ret)
        if show_one_layer:
            cv2.waitKey(0)
    if not show_one_layer:
        for idx, one_type in enumerate(all_ret):
            map_show = one_type[0] / 3 + one_type[1] / 3 + one_type[2] / 3
            show = map_show * 0.8 + img_ori * 0.2
            show = show.astype(np.uint8)
            map_show = map_show.astype(np.uint8)
            cv2.imshow(f"all_{SHOW_NAME[idx]}", show)
            cv2.imwrite('./cam_results/head_cont' + str(idx) + SHOW_NAME[idx] + ".jpg", show)
            # cv2.imshow(f"map_{SHOW_NAME[idx]}", map_show)
        cv2.waitKey(0)


ret = []
stride = [8, 16, 32, 64, 128]
efficientdet = EfficientDet()
path = 'img/3.jpg'
image = Image.open(path)
ret = efficientdet.detect_image(image)
# output_list = efficientdet.detect_image_fcos(image)
# for i, f in enumerate(output_list):
#     ret.append(f.reshape(1, stride[i], stride[i], 91))

# features1 = torch.randn(1,3,13,13,10)
# features2 = torch.randn(1,3,26,26,10)
# features3 = torch.randn(1,3,52,52,10)

show_CAM(path, ret, 2)