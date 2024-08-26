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

from pydantic import BaseModel

router = APIRouter()
ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
generate_url = f'{ollama_url}/api/generate'
chat_url = f'{ollama_url}/v1/chat/completions'

prompts_path = os.path.join(Path(__file__).parent.parent.parent.absolute(), 'prompts')
template_loader = jinja2.FileSystemLoader(searchpath=prompts_path)
template_env = jinja2.Environment(loader=template_loader)
vqa_template = template_env.get_template('vqa_transcribe.j2')
find_eval_input_template = template_env.get_template('get_name_evaluation_inputs.j2')
name_eval_template = template_env.get_template('get_name_evaluation.j2')

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
    injected_payload: Any = None


class ChatForJson(Chat):
    json_template: dict = {}


# endregion


# region helper functions
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
    for i in t.messages[::-1]:
        if i['role'] != 'system':
            continue
        i['content'] = prompt
        break
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

    return JSONResponse(output)


def dict_to_sorted_dict(dictionary):
    od = OrderedDict()
    for i in sorted(dictionary.keys()):
        od[i] = dictionary[i]
    return od
# endregion


@router.post('transcribe')
def transcribe(t: Generate):
    prompt = vqa_template.render({'transcription': t.prompt})
    payload = {
        'prompt': prompt,
        'model': 'gemma2-2b-Chinese'
    }

    return respond_helper(generate_url, payload, t.stream)


@router.post('chat/find_eval_input')
def find_eval_input(t: ChatForJson):
    prompt = find_eval_input_template.render()
    for i in t.messages[::-1]:
        if i['role'] != 'system':
            continue
        i['content'] = prompt
        break
    payload = t.model_dump(exclude={'injected_payload'})
    payload['stream'] = True

    return chat_for_json(payload, t.json_template)


@router.post('chat/name_eval')
def name_eval(t: Chat):
    name_eval_inputs = dict_to_sorted_dict(t.injected_payload)
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
        'is boy': '性别'
    }
    can_eval = True
    missing_info = []
    for v in name_eval_inputs.values():
        if v is None:
            can_eval = False
            missing_info.append(variable_map[v])
    if not can_eval:
        new_sys_msg = (
            f'用户似乎想要请南瓜道士帮忙评价名字，但是输入的信息不全。南瓜道士还需要以下信息：\n'
            f'{", ".join(missing_info)}'
            '可以请用户再次提供这些信息。'
        )
        t = conversation_injection(t, new_sys_msg)
        payload = t.model_dump(exclude={'injected_payload'})
        return respond_helper(generate_url, payload, True)
    hash_obj = sha256(json.dumps(name_eval_inputs).encode()).hexdigest()
    if hash_obj not in name_eval_results_cache:
        complete_inputs = prepare_payload(**name_eval_inputs)
        name_eval_results = get_name_eval(complete_inputs)
        name_eval_results_cache[hash_obj] = name_eval_results
    else:
        name_eval_results = name_eval_results_cache[hash_obj]
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
        'wai_ge': name_eval_results['外格→'].strip()[-1],
        'tian_ge': name_eval_results['天格→'][-1],
        'lunar': name_eval_results['出生农历'][-1],
        'bazi_score': name_eval_results['bazi_score'],
        'name_score': name_eval_results['name_score']
    }
    prompt = name_eval_template.render(context)
    t = conversation_injection(t, prompt)
    print(context)
    payload = t.model_dump(exclude={'injected_payload'})
    print(prompt)
    return respond_helper(chat_url, payload, t.stream)
