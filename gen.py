from argparse import ArgumentParser


class AppDelegate(object):

  def __init__(self):
    super().__init__()


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
