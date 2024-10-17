#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
pixi run python -u ./rag/retrieval/colbert_api/colbert_server.py \
    --config ./config/colbert_server/colbert_server_wiki2018.yaml
