#!/bin/bash
# Calibration of the v2 budgets on the production llama.cpp (sequential, slot-wait, 2 s pause).
cd ~/colibri-bench/quality || exit 1
python3 clients/qwen_llama_client.py --greedy --out calibration/qwen-greedy.jsonl
python3 clients/qwen_llama_client.py --samples 1 --out calibration/qwen-hermes.jsonl
curl -s 127.0.0.1:18080/health > calibration/health_after.txt
echo done > calibration/DONE
