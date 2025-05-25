# model_loader.py
"""
Handles loading the pre-trained Keras model and tokenizer.
"""
import streamlit as st
import tensorflow as tf
import pickle
import os
from config import MODEL_PATH, TOKENIZER_PATH

@st.cache_resource  # Cache the loaded model and tokenizer
def load_model_and_tokenizer():
    """
    Loads the pre-trained Keras model and tokenizer.
    Returns:
        tuple: (model, tokenizer, max_sequence_len) or (None, None, 0) if loading fails.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure it exists.")
        return None, None, 0
    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"Tokenizer file not found at {TOKENIZER_PATH}. Please ensure it exists.")
        return None, None, 0

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)

        # Infer max_sequence_len from the model's input layer
        # model.input_shape[1] is (sequence_len_param - 1) from training
        # The original full sequence length used for padding during training was model.input_shape[1] + 1
        if model.input_shape and len(model.input_shape) > 1 and model.input_shape[1] is not None:
            max_len_for_padding_logic = model.input_shape[1] + 1
        else:
            st.warning("Could not reliably infer max_sequence_len from the model's input shape. Using a default or previously set value if available.")
            # Fallback or raise an error if this is critical
            # For now, returning 0 and letting the main app handle it.
            return model, tokenizer, 0 # Or a default like 50 if you have one

        st.success("✅ Model and tokenizer loaded successfully!")
        return model, tokenizer, max_len_for_padding_logic
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None, 0

if __name__ == '__main__':
    # This part is for testing the loader independently
    # You would run this file directly: python model_loader.py
    print("Attempting to load model and tokenizer...")
    loaded_model, loaded_tokenizer, max_len = load_model_and_tokenizer()
    if loaded_model and loaded_tokenizer:
        print(f"Model: {type(loaded_model)}")
        print(f"Tokenizer: {type(loaded_tokenizer)}")
        print(f"Inferred MAX_SEQUENCE_LEN for padding logic: {max_len}")
        print(f"Model input shape expected (length of X): {loaded_model.input_shape[1]}")
    else:
        print("Failed to load model or tokenizer.")
