# Gemma-NameCraft
Do you have a new born baby and don't know how to name him/her in Chinese? Worry about wuxing or Fengshui? Looking for inspiration?

This project uses Gemma2 to assist new parents in naming their babies (in Chinese).

This project is part of Gemma Sprint 2024, an activity under the GDE program.

## What is it?

`Gemma-NameCraft` is a chatbot that assists new parents with Chinese name-giving.

Giving a name to a newborn baby in Chinese culture is important and difficult because it's believed to have a significant impact on their future. Parents often consider various beliefs, such as wuxing(五行) and bazi(八字), when choosing a name. Chinese names are typically composed of two or three characters, which can be difficult to choose and combine in a meaningful and harmonious way. Parents must also consider the stroke order and balance of the characters when selecting a name. Overall, naming a baby in Chinese culture is a complex and significant process that requires a lot of thought and consideration.

The chatbot can help new parents with these features:

1. Getting name inspiration from Chinese poetries (唐诗 or 宋词)
2. Getting name inspiration from the parents' unique experiences (a love story maybe?)
   1. Get name inspiration from images. Perhaps you have some very precious memories captured as photos, and want to use them to create unique names for your babies? We've got you covered!
3. Getting name inspiration from the parents' faith, worldview, expectation or hope for their children
4. Name evaluation:
   1. Evaluate the meaningfulness of each name
   2. Evaluate the wuxing or bazi of each name

**Disclaimer:** 

This is a project for showcasing Gemma's ability to create culturally sensitive contents. Gemma-NameCraft is provided for informational purposes only and is not intended to be a substitute for professional advice or judgment. The use of Gemma-NameCraft is at your own risk, and you are solely responsible for any consequences arising from the use of this tool.

By using Gemma-NameCraft, you agree to indemnify and hold harmless Gemma-NameCraft and its officers, directors, employees, and agents from any and all claims, damages, liabilities, costs, and expenses (including attorneys' fees) arising from or related to your use of Gemma-NameCraft.

Gemma-NameCraft is provided "as is" without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Gemma-NameCraft does not warrant that Gemma-NameCraft will meet your requirements or that it will be error-free, uninterrupted, or free from viruses or other harmful components.

In using Gemma-NameCraft, you acknowledge and agree that Gemma-NameCraft shall not be liable for any direct, indirect, incidental, special, or consequential damages arising from or related to your use of Gemma-NameCraft, including, but not limited to, damages for loss of profits, goodwill, use, data, or other intangible losses.

By using Gemma-NameCraft, you agree to these terms and conditions. If you do not agree to these terms and conditions, you should not use Gemma-NameCraft.

## How to run?

[![Alt text](https://img.youtube.com/vi/B4eiy7jpdyk/0.jpg)](https://www.youtube.com/watch?v=B4eiy7jpdyk)

Self-host using docker compose: first, create a `.env` file as shown below. 

```
POSTGRES_USER=<usr>
POSTGRES_PASSWORD=<pwd>
ollama_url=ollama:11434
backend_url=backend:5017
apis=apis:5023
healthEndPoint=http://backend:5017/health
host=postgres
paligemma_url=paligemma:5023
```

To use the flutter web app, go to [IshootLaser/Gemma-NameCraft-UI (github.com)](https://github.com/IshootLaser/Gemma-NameCraft-UI) and build the web app from there with `docker compose build`.

Under the project root, run `docker compose up`. Although cuda-compatible GPU is recommended for better performance, it is not required.



External Sources:

* [caoxingyu/chinese-gushiwen: 中华古诗文数据库和API。包含10000首古文(诗、词、歌、赋以及其它形式的文言文)，近4000名作者，10000名句 (github.com)](https://github.com/caoxingyu/chinese-gushiwen)
* [姓名测试打分生辰八字-起名字测试-测名字打分数算命-免费周易测算名字_起名网 (threetong.com)](https://www.threetong.com/ceming/baziceming/xingmingceshi.php)
* [callmefeifei/baby-names (github.com)](https://github.com/callmefeifei/baby-names)

```
Copyright © [2024] [Wei Zheng]. All rights reserved.
```
