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
