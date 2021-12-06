import numpy as np

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
