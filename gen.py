import os
from argparse import ArgumentParser, Namespace

import torch

from src.util import load_image, mksquare, save_image_tensor
from src.od_attack import YOLOv5VanishAttack


class AppDelegate(object):

  def __init__(self):
    super().__init__()

  def run(self, args: Namespace) -> int:
    if not os.path.exists(args.img_path):
      print(f'The image file path -i/--image=\'{args.img_path}\' is invalid.')
      return -1
    else:
      root_dir, ci = os.path.split(args.out_path if args.out_path is not None else os.getcwd())
      if '.' not in ci:
        img_file_name = os.path.split(args.img_path)[-1]
        img_file_name = img_file_name[:img_file_name.rindex('.')]
        args.out_path = os.path.join(root_dir, ci, f'{img_file_name}_adv.bmp')

      os.makedirs(os.path.split(args.out_path)[0], exist_ok=True)

    img_size = [args.img_size] * 2
    img = load_image(args.img_path)
    img = mksquare(img, size=img_size)

    # @todo(meo-s): change following two lines for users to use their own settings.
    yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval()
    adv_attack = YOLOv5VanishAttack(model=yolov5, conf_thres=0.15, iou_thres=0.9, alpha=0.0015, eps=0.015, max_iter=20)

    x = img.astype(float) / 255
    x_adv = (adv_attack(x, verbose=args.verbose) * 255).byte()

    save_image_tensor(args.out_path, x_adv)

    return 0


if __name__ == '__main__':
  arg_parser = ArgumentParser()
  arg_parser.add_argument('-i', '--image', dest='img_path', type=str, required=True,
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
  arg_parser.add_argument('-s', '--size', dest='img_size', type=int, default=640,
                          help='''
                          A image size that will be passed to model and created as adversarial example.''')
  arg_parser.add_argument('-v', '--verbose', dest='verbose', type=bool, default=True)

  args = arg_parser.parse_args()
  app = AppDelegate()
  exit(app.run(args))
