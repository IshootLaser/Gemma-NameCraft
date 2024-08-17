FROM huggingface/transformers-pytorch-gpu as paligemma-base
RUN pip install bitsandbytes accelerate

FROM paligemma-base as paligemma-jnb
RUN pip install notebook
EXPOSE 8888
# run jupyter notebook as entrypoint
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

FROM paligemma-base as paligemma-server
RUN pip install Flask==2.3.2 pillow==10.4.0
COPY ./paligemma_server /app
WORKDIR /app
RUN mkdir models
ENTRYPOINT python3 api.py

