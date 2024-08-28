import json
from hashlib import sha256
from typing import List, Dict, Any

import requests
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
import os
import jinja2
from pathlib import Path
from parsers.evaluate_name import prepare_payload, get_name_eval
from collections import OrderedDict
from utils.get_embeddings import rerank_from_infinity

from pydantic import BaseModel

from utils.vector_search import vector_search

# region variables
router = APIRouter()
ollama_url = os.environ.get('OLLAMA_URL', 'http://ollama:11434')
generate_url = f'{ollama_url}/api/generate'
chat_url = f'{ollama_url}/v1/chat/completions'

# prepare prompts
prompts_path = os.path.join(Path(__file__).parent.parent.parent.absolute(), 'prompts')
template_loader = jinja2.FileSystemLoader(searchpath=prompts_path)
template_env = jinja2.Environment(loader=template_loader)
vqa_template = template_env.get_template('vqa_transcribe.j2')
find_eval_input_template = template_env.get_template('get_name_evaluation_inputs.j2')
name_eval_template = template_env.get_template('get_name_evaluation.j2')
poetry_template = template_env.get_template('get_poetry_reference.j2')
cultural_template = template_env.get_template('get_cultural_reference.j2')
host_template = template_env.get_template('host.j2')
with open(os.path.join(prompts_path, 'intents.json'), 'r', encoding='utf-8') as fh:
    intents = json.load(fh)

name_eval_results_cache = {}


# region fastAPI models
class Generate(BaseModel):
    prompt: str
    stream: bool = False


class Chat(BaseModel):
    model: str = 'gemma2-2b-Chinese'
    stream: bool = False
    max_tokens: int = 1024
    messages: List[Dict[str, str]]
    options: Dict[str, Any] = {}
    injected_payload: Any = {}


class ChatForJson(Chat):
    json_template: dict = {}


# endregion


# region helper functions
def get_reference(prompt, template):
    search_text = template.render({'requirement': prompt})
    results = vector_search(search_text, 6, True, rerank_limit=2)
    return '\n'.join(results)


def name_eval(t: ChatForJson):
    user_prompt = t.messages[-1]['content']
    try:
        name_eval_inputs = dict_to_sorted_dict(find_eval_input(t))
    except:
        new_sys_msg = (
            f'用户输入的聊天信息是：\n{user_prompt}\n'
            '用户似乎想请南瓜道士用生辰八字判断一个名字的好坏，但没有提供相对应的信息。\n'
            '但是南瓜道士需要用户提供完整的生辰八字信息。\n'
            '请友好地向用户索取缺失信息，但不要引开话题。'
        )
        t = conversation_injection(t, new_sys_msg)
        payload = t.model_dump(exclude={'injected_payload'})
        return respond_helper(chat_url, payload, t.stream)
    print(f'name eval input: {name_eval_inputs}')
    variable_map = {
        '_last_name': '姓氏',
        '_first_name': '名字',
        '_year': '出生年份',
        '_month': '出生月份',
        '_day': '出生日期',
        '_hour': '出生小时',
        '_minute': '出生分钟',
        '_province': '省份',
        '_city': '城市',
        'is_boy': '性别'
    }
    can_eval = True
    missing_info = []
    for k, v in name_eval_inputs.items():
        if (v is None) or (v == ''):
            can_eval = False
            missing_info.append(variable_map[k])
    if not can_eval:
        new_sys_msg = (
            f'用户输入的聊天信息是：\n{user_prompt}\n'
            f'用户似乎想要请南瓜道士帮忙评价名字，但是输入的信息不全。南瓜道士还需要以下信息：\n'
            f'{", ".join(missing_info)}\n'
            '请友好地向用户索取缺失信息，但不要引开话题。'
        )
        t = conversation_injection(t, new_sys_msg)
        payload = t.model_dump(exclude={'injected_payload'})
        return respond_helper(chat_url, payload, t.stream)
    hash_obj = sha256(json.dumps(name_eval_inputs).encode()).hexdigest()
    if hash_obj not in name_eval_results_cache:
        complete_inputs = prepare_payload(**name_eval_inputs)
        name_eval_results = get_name_eval(complete_inputs)
        name_eval_results_cache[hash_obj] = name_eval_results
    else:
        name_eval_results = name_eval_results_cache[hash_obj]
    try:
        context = {
            'name': name_eval_results['命主姓名'],
            'year': name_eval_inputs['_year'],
            'month': name_eval_inputs['_month'],
            'day': name_eval_inputs['_day'],
            'hour': name_eval_inputs['_hour'],
            'minute': name_eval_inputs['_minute'],
            'province': name_eval_inputs['_province'],
            'city': name_eval_inputs['_city'],
            'sex': '男' if name_eval_inputs['is_boy'] else '女',
            'success': name_eval_results['cheng_gong_yun'],
            'fundamental': name_eval_results['ji_chu_yun'],
            'personal_trait': name_eval_results['personal_trait'],
            'social_trait': name_eval_results['she_jiao_yun'],
            'wuxing': name_eval_results['wuxing'],
            'zong_ge': name_eval_results['zong_ge'][-1],
            'ren_ge': name_eval_results['人格→'][-1],
            'di_ge': name_eval_results['地格→'][-1],
            'wai_ge': name_eval_results['外格→'][-1],
            'tian_ge': name_eval_results['天格→'][-1],
            'lunar': name_eval_results['出生农历'][-1],
            'bazi_score': name_eval_results['bazi_score'],
            'name_score': name_eval_results['name_score']
        }
    except:
        new_sys_msg = (
            f'用户输入的聊天信息是：\n{user_prompt}\n'
            f'用户似乎想要请南瓜道士帮忙评价名字，但是输入的信息无法计算。南瓜道士还需要以下信息：\n'
            f'名字，出生年月日，几点几分，出生省市，性别。\n'
            '请友好地向用户索取缺失信息，但不要引开话题。'
        )
        t = conversation_injection(t, new_sys_msg)
        payload = t.model_dump(exclude={'injected_payload'})
        return respond_helper(chat_url, payload, t.stream)
    print(f'name eval context: {context}')
    prompt = name_eval_template.render(context)
    t = conversation_injection(t, prompt)
    payload = t.model_dump(exclude={'injected_payload'})
    payload['messages'] = payload['messages'][-2:]
    return respond_helper(chat_url, payload, t.stream)


def identify_intent(prompt: str) -> int:
    available_intents = [x['intent_desc'] for x in intents]
    rerank_scores = rerank_from_infinity(prompt, available_intents)
    print(rerank_scores)
    intent_ix = find_argmax(rerank_scores)
    # give some extra weight to the conversation host
    if rerank_scores[intent_ix] < 0.4:
        intent_ix = 2
    intent_id = intents[intent_ix]['id']
    print(f'identified intent: {available_intents[intent_ix]}')
    return intent_id


def response_generator(r):
    for i in r.iter_lines():
        yield f'data: {i.decode("utf-8")}\n\n'


def respond_helper(url, payload, use_stream=False):
    if use_stream:
        payload['stream'] = True
        r = requests.post(url, json=payload, stream=True)
        return StreamingResponse(response_generator(r), media_type='text/event-stream')
    else:
        payload['stream'] = False
        r = requests.post(url, json=payload)
        return JSONResponse(r.json())


def conversation_injection(t: Chat, prompt: str):
    t.messages[-1]['content'] = prompt
    return t


def chat_for_json(payload, json_template):
    n_retries = 10
    output = {}
    for i in range(n_retries):
        # an early stopper that stops bad generation
        first_char = True
        r = requests.post(chat_url, json=payload, stream=True)
        response = ''
        for j in r.iter_lines():
            j = j.decode('utf-8')
            if not j:
                continue
            if j.replace('data: ', '').strip() == '[DONE]':
                break
            content = json.loads(j.replace('data: ', '').strip())['choices'][0]['delta']['content'].strip()
            if not content:
                continue
            if first_char and not content.startswith('{'):
                # bad format, stop early
                r.close()
                break
            first_char = False
            response += content
        try:
            output = json.loads(response)
            if json_template:
                for k, v in json_template.items():
                    assert k in output, f'{k} not found in response.'
                    if output[k] is not None:
                        # first, try to cast type
                        output[k] = type(v)(output[k])
                        # then, check if the type matches
                        msg = f'{k} type mismatch. Expected {type(v)}, got {type(output[k])}'
                        assert isinstance(output[k], type(v)), msg
            break
        except (json.JSONDecodeError, AssertionError):
            print(f'{i + 1}th retry failed. Generated text: {response}.Remaining retries: {n_retries - i - 1}')
            continue

    if not output:
        raise Exception(f'Failed to get valid response after {n_retries} retries.')

    return output


def find_eval_input(t: ChatForJson):
    prompt = find_eval_input_template.render({'user_input': t.messages[-1]['content']})
    t.messages[-1]['content'] = prompt
    payload = t.model_dump(exclude={'injected_payload'})
    payload['stream'] = True

    return chat_for_json(payload, t.json_template)


def dict_to_sorted_dict(dictionary):
    od = OrderedDict()
    for i in sorted(dictionary.keys()):
        od[i] = dictionary[i]
    return od


def find_argmax(_list):
    _min = min(_list)
    argmax = 0
    for n, i in enumerate(_list):
        if i >= _min:
            argmax = n
            _min = i
    return argmax
# endregion


@router.post('transcribe')
def transcribe(t: Generate):
    prompt = vqa_template.render({'transcription': t.prompt})
    payload = {
        'prompt': prompt,
        'model': 'gemma2-2b-Chinese',
        'sample': True
    }

    return respond_helper(generate_url, payload, t.stream)


@router.post('chat/find_eval_input')
def find_eval_input_wrapper(t: ChatForJson):
    return JSONResponse(find_eval_input(t))


@router.post('chat/name_eval')
def name_eval_wrapper(t: ChatForJson):
    return name_eval(t)


@router.post('chat/main')
def chat_main(t: ChatForJson):
    prompt = t.messages[-1]['content']
    if prompt.__contains__('<imageCaption>'):
        intent = -1
        caption, prompt = prompt.split('<imageCaption>')
        context = {'poetry': '', 'culture': '', 'image': caption, 'user_input': prompt}
    else:
        intent = identify_intent(prompt)
        context = {
            'poetry': get_reference(prompt, poetry_template) if intent == 0 else '',
            'culture': get_reference(prompt, cultural_template) if intent == 3 else '',
            'image': '',
            'user_input': prompt
        }
    print(f'prepared_context: {context}')
    new_prompt = host_template.render(context)
    if intent == 1:
        return name_eval(t)
    t = conversation_injection(t, new_prompt)
    return respond_helper(chat_url, t.model_dump(exclude={'injected_payload'}), t.stream)
