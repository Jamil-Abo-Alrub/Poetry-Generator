# config.py
"""
Configuration file for the Poetry Generation App.
Stores constants like file paths.
"""
import os

# --- File Paths ---
# Assuming the model and tokenizer are in the same directory as the scripts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "poetry_generator_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pickle")

# You could also store other constants here if needed, e.g., default temperature
DEFAULT_TEMPERATURE = 0.7
