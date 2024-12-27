from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.gigachat import GigaChatModel

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='gigachat not installed'),
    pytest.mark.anyio,
]


def test_init():
    m = GigaChatModel('GigaChat', api_key='foobar')
    assert m.name() == 'sber:GigaChat'
