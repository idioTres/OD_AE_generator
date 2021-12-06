import pytest

from .. import util


def test_query_dict_without_default_value():
  assert util.query_dict({}, '') is None


def test_query_dict_with_default_value():
  default_value = 'default_value'
  assert util.query_dict({}, '', default_value) == default_value
