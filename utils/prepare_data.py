import os
from pathlib import Path
import subprocess
from time import time

from sqlalchemy.dialects.postgresql import insert

from utils.sql_tables import db_manager, Embeddings_bge_m3
from utils.get_embeddings import embed_from_infinity, rerank_from_infinity
from utils.completion import generate_completion
import json
from tqdm import tqdm
import hashlib
from traceback import print_exc


data_save_dir = os.environ.get('DATA_SAVE_DIR', os.path.join(Path(__file__).parent.parent.absolute(), './data'))
os.chdir(data_save_dir)
chinese_ancient_text_repo = 'https://github.com/caoxingyu/chinese-gushiwen.git'
chinese_dictionary_repo = 'https://github.com/pwxcoo/chinese-xinhua.git'
for i in [chinese_ancient_text_repo, chinese_dictionary_repo]:
    repo_name = i.split('/')[-1].split('.')[0]
    if os.path.isdir(os.path.join(data_save_dir, repo_name)):
        continue
    subprocess.run(['git', 'clone', i])

# prepare poetry
poetry_path = os.path.join(data_save_dir, 'chinese-gushiwen/guwen')
poetry_files = [os.path.join(poetry_path, x) for x in os.listdir(poetry_path) if x.endswith('.json')]
# prepare sentence
sentence_path = os.path.join(data_save_dir, 'chinese-gushiwen/sentence')
sentence_files = [os.path.join(sentence_path, x) for x in os.listdir(sentence_path) if x.endswith('.json')]
# prepare words
words_path = os.path.join(data_save_dir, 'chinese-xinhua/data/word.json')
name_character_candidates_path = os.path.join(data_save_dir, 'common_words_in_name.json')


def get_name_characters():
    with open(name_character_candidates_path, 'r', encoding='utf-8') as f:
        _ = json.load(f)
    _name_characters = []
    for v in _.values():
        _name_characters += v
    return set(_name_characters)


name_characters = get_name_characters()
# prepare idiom
idiom_path = os.path.join(data_save_dir, 'chinese-xinhua/data/idiom.json')
with db_manager.get_session() as session:
    uuid_list = session.query(Embeddings_bge_m3.uuid).all()
    uuid_set = {x[0] for x in uuid_list}


def poetry_sentence_generator(path_list):

    def helper(n=10000):
        for file in path_list:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for counter, line in enumerate(lines):
                    if counter >= n:
                        return
                    yield json.loads(line)

    return helper


def poetry_prepare_callback(poetry):
    if 'translation' in poetry:
        translation = poetry['translation']
    else:
        prompt = (
            f'请将古诗《{poetry["title"].strip()}》的原文翻译成白话文。\n'
            f'古诗的作者是{poetry["writer"].strip()}, 创作朝代是{poetry["dynasty"].strip()}。\n'
            f'古诗的原文是:\n{poetry["content"].strip()}\n'
            f'你的回答只应该包含翻译的内容，不要包含原文，不要额外写多余的文字。举例：\n'
            f'原文: 昔人已乘黄鹤去，此地空余黄鹤楼。\n'
            f'翻译:\n###\n古人已经乘坐黄鹤飞走了，这里只剩下黄鹤楼。\n###\n'
            '你的回复：\n'
        )
        translation = generate_completion(prompt, '你是中国古文学的大师。请根据用户的要求翻译古诗。只用中文回答。')

    doc = (
        f'古诗《{poetry["title"].strip()}》的作者是{poetry["writer"].strip()}, 创作朝代是{poetry["dynasty"].strip()}。'
        f'古诗的原文是:\n'
        f'{poetry["content"].strip()}\n'
        f'古诗翻译成白话文是: {translation}\n'
    )
    return {'raw_text': doc}


def sentence_prepare_callback(sentence):
    doc = (
        f'中国有句古代名句: {sentence["name"].strip()}\n'
        f'这句名句的出处是: {sentence["from"]}'
    )
    md5_hash = hashlib.md5(doc.encode()).hexdigest()
    return {'raw_text': doc, 'uuid': md5_hash}


def dictionary_generator(path):
    with open(path, 'r', encoding='utf-8') as f:
        words = json.load(f)
    words = [x for x in words if set(x['word']).intersection(name_characters)]
    word_count = len(words)
    print('total words:', word_count)

    def helper(n=2048):
        for counter, word in enumerate(words):
            if counter >= n:
                return
            yield word

    return helper


def word_prepare_callback(word):
    for k in ('word', 'explanation', 'strokes', 'more'):
        assert k in word, 'word format error'
    doc = (
        f'在新华字典里，字{word["word"]}的解释是: {word["explanation"]}\n'
        f'这个字的笔画数是: {word["strokes"]}\n'
        '' if word['more'].startswith('搜索') else f'关于这个字，还有更多的信息: {word["more"]}\n'
    )
    return {'raw_text': doc}


def idiom_prepare_callback(idiom):
    for k in ('derivation', 'example', 'explanation', 'word'):
        assert k in idiom, 'idiom format error'
    doc = (
        f'中国有一句古代流传下来的成语/短语 {idiom["word"]}。它的起源是 {idiom["derivation"]}。\n'
        f'它的意思是：{idiom["explanation"]}。\n'
        f'它曾被用在这个文献里：{idiom["example"]}\n'
    )
    return {'raw_text': doc}


def insert_helper(
        content_generator, content_prepare_callback, embedding_batch_size=4, insert_batch_size=32, text_samples=128,
        desc='processing ancient text'
):

    def update_embedding(_embedding_batch):
        embeddings = embed_from_infinity(text=[x['raw_text'] for x in _embedding_batch])
        for j, embedding in enumerate(embeddings):
            embedding_batch[j]['embedding'] = embedding
        return embedding_batch

    def bulk_insert(_insert_batch):
        with db_manager.get_connection() as conn:
            stmt = insert(Embeddings_bge_m3).values(_insert_batch)
            stmt = stmt.on_conflict_do_nothing(index_elements=['uuid'])
            conn.execute(stmt)
            conn.commit()

    embedding_batch = []
    insert_batch = []
    for content in tqdm(content_generator(text_samples), desc=desc, total=text_samples):
        uuid = hashlib.md5(str(content).encode()).hexdigest()
        if uuid in uuid_set:
            continue
        try:
            item = content_prepare_callback(content)
            item['uuid'] = uuid
            if len(item['raw_text']) > 2014:
                continue
        except Exception as e:
            print_exc()
            continue
        embedding_batch.append(item)

        if len(embedding_batch) >= embedding_batch_size:
            try:
                embedding_batch = update_embedding(embedding_batch)
            except Exception as e:
                print_exc()
                embedding_batch = []
                continue
            insert_batch += embedding_batch
            embedding_batch = []
        if len(insert_batch) >= insert_batch_size:
            try:
                bulk_insert(insert_batch)
            except Exception as e:
                print_exc()
                insert_batch = []
                continue
            insert_batch = []

    if embedding_batch:
        embedding_batch = update_embedding(embedding_batch)
        insert_batch += embedding_batch

    if insert_batch:
        bulk_insert(insert_batch)
    return


if __name__ == '__main__':
    insert_helper(
        poetry_sentence_generator(poetry_files), poetry_prepare_callback, text_samples=10000, desc='processing poetry'
    )
    insert_helper(
        poetry_sentence_generator(sentence_files), sentence_prepare_callback, text_samples=10000,
        desc='processing sentence'
    )
    insert_helper(
        dictionary_generator(words_path), word_prepare_callback, text_samples=2324, desc='processing words'
    )
    insert_helper(
        dictionary_generator(idiom_path), idiom_prepare_callback, text_samples=28744, desc='processing idioms'
    )
    # run a test
    from sqlalchemy import select
    test_str = '李白写过哪些关于月亮和喝酒的诗句？'
    test_embedding = embed_from_infinity(test_str)
    with db_manager.get_session() as session:
        out = session.scalars(
            select(Embeddings_bge_m3).order_by(Embeddings_bge_m3.embedding.l2_distance(test_embedding[0])).limit(5)
        )
        results = [x.raw_text for x in out]
        out = rerank_from_infinity(test_str, results)
        print('Done')
