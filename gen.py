import os
from typing import Any, Dict
from argparse import ArgumentParser, Namespace

import cv2
import torch

from src.util import mksquare, save_image_tensor
from src.od_attack import YOLOv5VanishAttack


class AppDelegate(object):

  def __init__(self):
    super().__init__()

  def run(self, args: Namespace) -> int:
    if not os.path.exists(args.image_path):
      print(f'The image file path -i/--image=\'{args.image_path}\' is invalid.')
      return -1
    else:
      root_dir, ci = os.path.split(args.out_path if args.out_path is not None else os.getcwd())
      if '.' not in ci:
        image_file_name = os.path.split(args.image_path)[-1]
        image_file_name = image_file_name[:image_file_name.rindex('.')]
        args.out_path = os.path.join(root_dir, ci, f'{image_file_name}_adv.bmp')

      os.makedirs(os.path.split(args.out_path)[0], exist_ok=True)

    size = [32 * 20] * 2  # @todo(meo-s): change magic number to program argument.
    img = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = mksquare(img, size=size)

    # @todo(meo-s): change following two lines for users to use their own settings.
    yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval()
    adv_attack = YOLOv5VanishAttack(model=yolov5, conf_thres=0.15, iou_thres=0.9, alpha=0.0015, eps=0.015, max_iter=20)

    x = img.astype(float) / 255
    x_adv = (adv_attack(x) * 255).byte()

    save_image_tensor(args.out_path, x_adv)

    return 0


if __name__ == '__main__':
  arg_parser = ArgumentParser()
  arg_parser.add_argument('-i', '--image', dest='image_path', type=str, required=True,
                          help='''
                          A path to source image which will be used to generate adversarial example.''')
  arg_parser.add_argument('-m', '--model', dest='model_type', type=str,
                          help='''
                          A type of model which is victim of adversarial attack.
                          It it the reserved argument, so regardless of its value, YOLOv5 is used to generate adversarial example.''')
  arg_parser.add_argument('-w', '--weights', dest='weights_path', type=str,
                          help='''
                          A path to weights file of model which is victim of adversarial attack.''')
  arg_parser.add_argument('-o', '--out', dest='out_path', type=str,
                          help='''
                          A path to save generated adversarial example.
                          If not given, cwd is used as default output directory.''')

  args = arg_parser.parse_args()
  app = AppDelegate()
  exit(app.run(args))
