from utils.sql_tables import db_manager, Embeddings_bge_m3
from utils.get_embeddings import embed_from_infinity, rerank_from_infinity
from sqlalchemy import select


def vector_search(_input: str, limit=32, rerank=False, rerank_limit=None):
    if rerank_limit is None:
        rerank_limit = limit
    embedding = embed_from_infinity(_input)
    with db_manager.get_session() as session:
        result = session.scalars(
            select(Embeddings_bge_m3).order_by(Embeddings_bge_m3.embedding.l2_distance(embedding[0])).limit(limit)
        ).all()
    if not rerank:
        return [x.raw_text for x in result]
    rerank_score = rerank_from_infinity(_input, [x.raw_text for x in result])
    pair = ((a, b) for (a, b) in zip([x.text for x in result], rerank_score))
    reranked = sorted(pair, key=lambda x: x[1], reverse=True)[:rerank_limit]
    return [x[0] for x in reranked]
