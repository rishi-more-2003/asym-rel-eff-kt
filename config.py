import os
from dotenv import load_dotenv

load_dotenv()

TINKER_API_KEY = os.getenv("TINKER_API_KEY", "")

# Qwen3-8B-Base is the child agent's starting checkpoint (Section 3.2/3.3)
BASE_MODEL = "Qwen/Qwen3-8B-Base"

# Qwen3-8B hybrid is the caregiver + child runtime model
AGENT_MODEL = "Qwen/Qwen3-8B"

# Qwen3-235B-A22B-Instruct for high-quality dataset generation (MoE, 22B active)
GENERATION_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

NUM_TRAINING_TASKS = 160
NUM_EVAL_TASKS = 40
TOTAL_TASKS = NUM_TRAINING_TASKS + NUM_EVAL_TASKS
OVERGENERATE_FACTOR = 1.15  # generate ~15% extra to allow filtering

DIFFICULTY_LEVELS = list(range(1, 9))  # 1-step to 8-step
TASKS_PER_DIFFICULTY = TOTAL_TASKS // len(DIFFICULTY_LEVELS)  # 25 per level

TASK_CATEGORIES = [
    "meal_preparation",
    "laundry",
    "cleaning",
    "furniture_assembly",
    "gardening",
    "personal_care",
    "pet_care",
    "home_repair",
    "organizing",
    "cooking_baking",
]

ROOM_CATEGORY_MAP = {
    "meal_preparation": ["kitchen"],
    "laundry": ["laundry_room", "bathroom", "bedroom"],
    "cleaning": ["kitchen", "bathroom", "living_room", "bedroom"],
    "furniture_assembly": ["living_room", "bedroom", "garage"],
    "gardening": ["garden", "garage"],
    "personal_care": ["bathroom", "bedroom"],
    "pet_care": ["kitchen", "living_room", "garden"],
    "home_repair": ["garage", "kitchen", "bathroom", "living_room"],
    "organizing": ["bedroom", "living_room", "kitchen", "garage"],
    "cooking_baking": ["kitchen"],
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TASKS_DIR = os.path.join(DATA_DIR, "tasks")
TASK_DB_PATH = os.path.join(DATA_DIR, "task_database.json")
ONTOLOGY_PATH = os.path.join(DATA_DIR, "object_ontology.json")
SKELETONS_PATH = os.path.join(DATA_DIR, "task_skeletons.json")

SAMPLING_TEMPERATURE = 0.7
SAMPLING_MAX_TOKENS = 3072

BM25_DUPLICATE_THRESHOLD = 0.7

# ── Experiment Pipeline ──────────────────────────────────────────────
CAREGIVER_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
CHILD_MODEL = "Qwen/Qwen3-8B"

LORA_RANK = 16
LORA_LR = 2e-5

WORKING_MEMORY_SIZE = 8
MAX_EPISODE_TURNS = 15

SALIENCE_ALPHA = 0.3
SALIENCE_BETA = 0.4
SALIENCE_GAMMA = 0.3
SALIENCE_TAU = 0.3

SCAFFOLDING_WINDOW = 5
SCAFFOLDING_THRESHOLDS = (0.3, 0.7)

ACTION_MATCH_THRESHOLD = 0.5
LORA_BATCH_SIZE = 4

NUM_SEEDS = 3
NUM_EPISODES = 160
CHECKPOINT_EVERY = 20

# Parallelism modes (pick one):
#   "combined"    = judge+caregiver in one 235B call (fastest, default)
#   "speculative" = fire judge + 2 caregiver variants in parallel (more $ but still fast)
#   "sequential"  = judge, then caregiver (slowest, cheapest)
CAREGIVER_MODE = "combined"

MAX_CONCURRENT_RUNS = 12

# Adaptive turn limits: scale max turns by difficulty to avoid wasting
# budget on easy tasks that should finish quickly.
ADAPTIVE_TURN_LIMITS = True
TURN_LIMIT_BY_DIFFICULTY = {
    1: 4,  2: 6,  3: 8,  4: 10,
    5: 12, 6: 13, 7: 14, 8: 15,
}

RUNS_DIR = os.path.join(DATA_DIR, "runs")
