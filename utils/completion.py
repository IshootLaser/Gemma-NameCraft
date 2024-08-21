import os
import requests

ollama_url = os.environ.get('OLLAMA_URL', 'http://ollama:11434')
default_sys_msg = 'You are an AI agent. Faithfully respond to user\'s conversation.'


def generate_completion(
        prompt: str, system_msg: str = default_sys_msg, model: str = 'gemma2-2b-Chinese', url: str = ollama_url
):
    url += '/api/generate'
    payload = {
        'model': model,
        'prompt': prompt,
        'system': system_msg,
        'stream': False,
        'options': {
            'num_ctx': 8192
        }
    }
    r = requests.post(url, json=payload)
    if r.status_code != 200:
        raise Exception(f'Failed to complete text: {r.json()}')
    return r.json()['response']


def generate_chat_completion(
        prompt: str, system_msg: str = default_sys_msg, model: str = 'gemma2-2b-Chinese', url: str = ollama_url,
        max_tokens: int = 256
):
    url += '/v1/chat/completions'
    payload = {
        'model': model,
        'stream': False,
        'max_tokens': max_tokens,
        'messages': [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': prompt}
        ]
    }
    r = requests.post(url, json=payload)
    if r.status_code != 200:
        raise Exception(f'Failed to complete text: {r.json()}')
    return r.json()['choices'][0]['message']['content']


# simple tests
if __name__ == '__main__':
    _prompt = '你好'
    _completion = generate_completion(_prompt)
    print('complete', _completion)
    _completion = generate_chat_completion(_prompt, system_msg='你是一个主要说中文的AI助手。和用户愉快的聊天吧！', max_tokens=20)
    print('complete from chat', _completion)
