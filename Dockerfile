FROM huggingface/transformers-pytorch-gpu as paligemma-base
RUN pip install bitsandbytes accelerate

FROM paligemma-base as paligemma-jnb
RUN pip install notebook
EXPOSE 8888
# run jupyter notebook as entrypoint
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

FROM paligemma-base as paligemma-server
RUN pip install fastapi[standard] pillow==10.4.0 Flask==3.0.3
COPY ./models/paligemma_4bit /app/models/paligemma
COPY ./apis/ /app
WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENTRYPOINT fastapi run paligemma_fastAPI.py --port 5023
#ENTRYPOINT python3 paligemma_api.py

FROM ollama/ollama as ollama-builder
RUN mkdir app
COPY ./models/Gemma-2-2b-Chinese-it-GGUF /app/Gemma-2-2b-Chinese-it-GGUF
COPY ./models/Modelfile /app/Modelfile
COPY ./models/build_ollama.sh /app/build_ollama.sh
WORKDIR /app
RUN chmod +x build_ollama.sh && ./build_ollama.sh
RUN rm -rf Gemma-2-2b-Chinese-it-GGUF
RUN cp -r ~/.ollama /tmp

FROM ollama/ollama as ollama-gemma
RUN mkdir app
COPY ./models/run_gemma.sh /app/run_gemma.sh
WORKDIR /app

COPY --from=ollama-builder /tmp/.ollama /root/.ollama
RUN chmod +x run_gemma.sh