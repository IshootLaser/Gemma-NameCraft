import requests
import json

url = "http://localhost:11434/v1/chat/completions"

payload = json.dumps({
  "model": "gemma2-2b-Chinese",
  "stream": True,
  "max_tokens": 5,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ]
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload, stream=True)
for line in response.iter_lines():
  if line:
    print(line)

print(response.text)
