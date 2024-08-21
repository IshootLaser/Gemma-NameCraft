from sqlalchemy import create_engine, text, Integer, Index, JSON, Text, select
from sqlalchemy.orm import sessionmaker, mapped_column, declarative_base
import os
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, inspect


class DBManager:
    _instances = {}
    host = os.environ.get('host', '127.0.0.1')
    port = os.environ.get('port', '5432')
    database = os.environ.get('database', 'embeddings')
    user = os.environ.get('POSTGRES_USER', 'usr')
    password = os.environ.get('POSTGRES_PASSWORD', 'pwd')
    conn_string = f'postgresql+psycopg2://{user}:{password}@{host}:{port}'
    engine = None
    base = None

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(DBManager, cls).__new__(cls)
            engine = create_engine(cls.conn_string, isolation_level='AUTOCOMMIT')
            with engine.connect() as conn:
                result = conn.execute(text('SELECT datname FROM pg_database WHERE datistemplate = false'))
                result = [x[0] for x in result]
                if cls.database not in result:
                    conn.execute(text(f'CREATE DATABASE {cls.database}'))
            cls.engine = create_engine(f'{cls.conn_string}/{cls.database}')
            with cls.engine.connect() as conn:
                conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
                conn.execute(text("SET client_encoding = 'UTF-8'"))
                conn.commit()
            cls.base = declarative_base()
            cls.base.metadata.create_all(cls.engine)
        return cls._instances[cls]

    @classmethod
    def get_session(cls):
        return sessionmaker(bind=cls.engine)()

    @classmethod
    def recreate_all(cls):
        cls.base.metadata.create_all(cls.engine)

    @classmethod
    def get_connection(cls):
        return cls.engine.connect()

    @classmethod
    def get_constrainsts(cls, table_name):
        inspector = inspect(cls.engine)
        return inspector.get_unique_constraints(table_name)


db_manager = DBManager()


class Embeddings_bge_m3(db_manager.base):
    __tablename__ = 'embeddings_bge_m3'
    id = Column(Integer, primary_key=True)
    embedding = mapped_column(Vector(1024))
    raw_text = Column(JSON)
    uuid = Column(Text, unique=True, index=True)


index_name = 'embedding_index'
index = Index(
    index_name,
    Embeddings_bge_m3.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'vector_l2_ops'}
)

db_manager.recreate_all()
with db_manager.get_connection() as conn:
    result = conn.execute(text("SELECT tablename, indexname FROM pg_indexes WHERE schemaname = 'public'"))
    result = [(x[0], x[1]) for x in result]
    if (Embeddings_bge_m3.__tablename__, index_name) not in result:
        index.create(db_manager.engine)
# some test
# item1 = Embeddings_bge_m3(embedding=[1, 2, 3])
# item2 = Embeddings_bge_m3(embedding=[3, 1, 3])
# with db_manager.get_session() as session:
#     session.add(item1)
#     session.add(item2)
#     session.commit()
#     print('Item added to the database.')
#     out = session.scalars(
#         select(Embeddings_bge_m3).order_by(Embeddings_bge_m3.embedding.l2_distance([3, 1, 2])).limit(5)
#     )
#     print('Done')
