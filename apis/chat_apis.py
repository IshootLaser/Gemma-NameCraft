# from flask import Flask, request, Response, stream_with_context
# import requests
# from utils.get_embeddings import embed_from_infinity, rerank_from_infinity
#
# app = Flask(__name__)
#
#
# def identify_intent(prompt: str):
#     intent_list = []
#     intent_desc = []
#     result = rerank_from_infinity(prompt, intent_desc)
#     argmax = 0
#     max_score = -99
#     n = 0
#     for n, i in enumerate(result):
#         if i > max_score:
#             max_score = i
#             argmax = n
#     return intent_list[n]
#
#
# @app.route('/')
# def
#
#
#
#
# app.run('0.0.0.0', 5077)
