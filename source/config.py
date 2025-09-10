from pathlib import Path

# Store useful variables and configuration
project_root = Path(__file__).parent.parent
CONFIGS_DIR = project_root / 'config'
MODELS_DIR = project_root / 'models'
SAVED_MODELS = MODELS_DIR / 'saved'

DATA = project_root / 'data'
RAW_DATA = DATA / 'raw'
PROCESSED_DATA = DATA / 'processed'
FINAL_DATA = DATA / 'final'

CACHE_DIR = MODELS_DIR / 'saved' / 'huggingface_cache'

REPORTS_FIGURES_DIR = project_root / 'reports' / 'figures'

CLASS_NAMES = ['margarine', 'bread', 'cheese', 'pan', 'sandwich']
NUM_CLASSES = 5
