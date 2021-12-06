import pytest

from .. import util


def test_query_dict_without_default_value():
  assert util.query_dict({}, '') is None
