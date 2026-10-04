from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "datasets"
RAW_DATA_DIR = DATA_ROOT / "voxlingua107"
WAV_DATA_DIR = DATA_ROOT / "voxlingua107_wav"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# LANGUAGE CONFIGURATION
# ============================================================

LANGUAGES = [
    "en",
    "hi",
    "te",
    "ta",
    "bn",
    "gu",
    "mr",
    "fr",
    "de",
    "es",
]

NUM_LANGUAGES = len(LANGUAGES)

LANGUAGE_TO_ID = {
    language: index
    for index, language in enumerate(LANGUAGES)
}

ID_TO_LANGUAGE = {
    index: language
    for language, index in LANGUAGE_TO_ID.items()
}


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
NUM_CHANNELS = 1


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

N_MELS = 80
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160


# ============================================================
# DATA SPLIT CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42
