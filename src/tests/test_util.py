import os
import shutil
import hashlib

import cv2
import torch
import numpy as np
import pytest

from .. import util


def test_query_dict():
  key = 'key'
  some_dict = dict([[key, 'value']])
  assert util.query_dict(some_dict, key, 'not this') == some_dict[key]

  '''
  with default value
  '''
  default_value = 'default_value'
  assert util.query_dict({}, '', default_value) == default_value

  '''
  without default value
  '''
  assert util.query_dict({}, '') is None


def test_load_image():
  img = np.random.randint(low=0, high=255, size=(10, 10, 3), dtype=np.uint8)

  '''
  Create unique temporary directory.
  '''
  md5 = hashlib.md5(img.reshape(-1))
  test_dir = f'.{md5.hexdigest()}'
  os.mkdir(test_dir)

  test_img_path = os.path.join(test_dir, 'test.png')
  cv2.imwrite(test_img_path, img[:, :, ::-1])  # load_image() automatically converts BGR format to RGB format.

  ret = util.load_image(test_img_path)
  shutil.rmtree(test_dir)
  assert np.all(ret == img)

  assert util.load_image(test_img_path) is None


def test_mksquare():
  for initial_size in [(10, 1), (1, 10)]:
    img = np.zeros(initial_size)
    img = util.mksquare(img, pad_color=(0, 0, 0), size=(100, 100))
    assert np.all(img == np.zeros((100, 100)))

  pad_color = (255, )
  img = np.zeros((4, 8))
  img = util.mksquare(img, pad_color)

  pad = np.full((2, 8), fill_value=pad_color)
  target_img = np.concatenate([pad, np.zeros((4, 8)), pad], axis=0)

  assert np.all(img == target_img)


def test_xywh2xyxy():
  xywh, target = torch.LongTensor([1, 4, 1000, 2]), torch.LongTensor([-499, 3, 501, 5])
  xyxy = util.xywh2xyxy(xywh)
  assert torch.all(xyxy == target).item() is True


def test_save_image_tensor():
  img = torch.randint(low=0, high=256, size=(10, 10, 3), dtype=torch.uint8)

  md5 = hashlib.md5(img.numpy().reshape(-1))
  test_img_path = f'.{md5.hexdigest()}'

  with pytest.raises(ValueError, match=r'^.*image extension.*$'):
    util.save_image_tensor(f'{test_img_path}.fail', img)

  with pytest.raises(ValueError, match=r'^.*dtype is byte.*$'):
    invalid_input = img.long()
    util.save_image_tensor(f'{test_img_path}.bmp', invalid_input)

  with pytest.raises(ValueError, match=r'^.*tensor.*invalid shape.*$'):
    invalid_input = img.permute(0, 2, 1)
    util.save_image_tensor(f'{test_img_path}.bmp', invalid_input)

  img = img[None, None, ...]
  util.save_image_tensor(f'{test_img_path}.bmp', img)
  assert os.path.exists(f'{test_img_path}.bmp')
  os.remove(f'{test_img_path}.bmp')
