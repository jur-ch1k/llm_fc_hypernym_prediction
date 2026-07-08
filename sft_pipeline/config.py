from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = "datasets/context_analyser_dataset.tsv"
FASTTEXT_PATH = "examples/fasttext_baseline.json"
CORPUS_DIR = "corpus/train_eval_texts"
WORDNET_PATH = "wordnets/RuWordNet"
SYSTEM_PROMPT_PATH = "prompts/system_sft.md"
OUTPUT_TRAJECTORIES = "output/trajectories"
OUTPUT_DATASET = "output/dataset"
FASTTEXT_TOP_K = 3
VIRTUAL_ROOT = "null"
POS = "N"
MAX_WORDS = 3
MAX_HYPONYMS = 5
