"""Entry point: generate the household task database.

Usage:
    1. Fill in your Tinker API key in .env
    2. pip install -r requirements.txt
    3. python run_generate_tasks.py

This runs a multi-stage pipeline (ontology -> skeletons -> expansion ->
verification -> filtering) using Qwen3-235B-A22B-Instruct via Tinker,
producing data/task_database.json with ~200 structured household scenarios
(160 train / 40 eval) across 8 difficulty levels.
"""

from src.generate_tasks import main

if __name__ == "__main__":
    main()
