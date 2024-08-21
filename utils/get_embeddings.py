import os
from typing import List, Union

import requests

infinity_server = os.environ.get('INFINITY_SERVER', 'http://embedding_services:7997')


def embed_from_infinity(
        text: Union[str, List[str]], model: str = 'BAAI/bge-m3', url: str = infinity_server,
        batch_limit: int = 4
) -> List[List[float]]:
    url += '/embeddings'
    if (len(text) > batch_limit) and isinstance(text, list):
        raise ValueError('Too many texts to embed at once')
    payload = {
        'input': [text] if isinstance(text, str) else text,
        'model': model,
        'user': 'string'
    }
    r = requests.post(url, json=payload)
    if r.status_code != 200:
        raise Exception(f'Failed to embed text: {r.json()}')
    embeddings = [(x['index'], x['embedding']) for x in r.json()['data']]
    # sort by index
    embeddings = [x[1] for x in sorted(embeddings, key=lambda x: x[0])]
    return embeddings


def rerank_from_infinity(
        query: str, candidates: List[str], model: str = 'BAAI/bge-reranker-base', url: str = infinity_server
):
    url += '/rerank'
    assert len(candidates) > 0, 'No candidates to rerank'
    payload = {
        'query': query,
        'documents': candidates,
        'return_documents': False,
        'model': model
    }
    r = requests.post(url, json=payload)
    results = [(x['index'], x['relevance_score']) for x in r.json()['results']]
    return [x[1] for x in sorted(results, key=lambda x: x[0])]


# simple tests
if __name__ == '__main__':
    _text = ['我爱北京天安门', '天安门上太阳升']
    _embeddings = embed_from_infinity(_text)
    print(_embeddings)
    assert len(_embeddings) == 2
    _text = '我爱北京天安门'
    _embeddings = embed_from_infinity(_text)
    print(_embeddings)
    assert len(_embeddings) == 1
