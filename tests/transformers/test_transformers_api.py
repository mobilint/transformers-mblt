from pprint import pprint

from transformers_mblt.utils import list_models, list_tasks


def test_transformers_api():
    print(list_tasks())
    pprint(list_models())
