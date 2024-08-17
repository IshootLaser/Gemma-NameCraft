#!/bin/bash

ollama serve &
ollama list
ollama run gemma2-2b-Chinese
wait $pid