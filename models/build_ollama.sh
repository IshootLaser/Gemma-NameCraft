#!/usr/bin/env bash

ollama serve &
ollama list
ollama create "gemma2-2b-Chinese" -f Modelfile