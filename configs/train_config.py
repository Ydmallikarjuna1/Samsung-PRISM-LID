from configs.config import (
    SEED,
    NUM_LANGUAGES,
    SAMPLE_RATE,
)


# ============================================================
# DATA
# ============================================================

TRAIN_MANIFEST = "datasets/manifests/train.jsonl"
VALID_MANIFEST = "datasets/manifests/valid.jsonl"
TEST_MANIFEST = "datasets/manifests/test.jsonl"

SAMPLE_RATE = SAMPLE_RATE

MAX_DURATION = 3.0


# ============================================================
# DATALOADER
# ============================================================

BATCH_SIZE = 4

NUM_WORKERS = 0

PIN_MEMORY = False


# ============================================================
# MODEL
# ============================================================

INPUT_DIM = 80

CHANNELS = 512

EMBEDDING_DIM = 192

NUM_CLASSES = NUM_LANGUAGES


# ============================================================
# AAM-SOFTMAX
# ============================================================

AAM_MARGIN = 0.2

AAM_SCALE = 30.0


# ============================================================
# OPTIMIZATION
# ============================================================

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 10


# ============================================================
# TRAINING
# ============================================================

SEED = SEED

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

GRADIENT_CLIP = 5.0


# ============================================================
# CHECKPOINTS
# ============================================================

SAVE_EVERY_EPOCH = True

CHECKPOINT_DIR = "checkpoints"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def print_config():
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    print(f"Device          : {DEVICE}")
    print(f"Sample rate     : {SAMPLE_RATE}")
    print(f"Max duration    : {MAX_DURATION}s")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Num workers     : {NUM_WORKERS}")

    print(f"Input dimension : {INPUT_DIM}")
    print(f"Channels        : {CHANNELS}")
    print(f"Embedding dim   : {EMBEDDING_DIM}")
    print(f"Num classes     : {NUM_CLASSES}")

    print(f"AAM margin      : {AAM_MARGIN}")
    print(f"AAM scale       : {AAM_SCALE}")

    print(f"Learning rate   : {LEARNING_RATE}")
    print(f"Weight decay    : {WEIGHT_DECAY}")
    print(f"Epochs          : {NUM_EPOCHS}")

    print("=" * 60)


if __name__ == "__main__":
    print_config()