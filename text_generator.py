# text_generator.py
"""
Contains the logic for generating text using the trained model.
"""
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st # For st.warning, st.error

def generate_single_word(model_loaded, tokenizer_loaded, current_sequence_tokens, max_model_input_len, temperature):
    """Helper to predict a single next word."""
    if not current_sequence_tokens:
        st.warning("Cannot predict next word from empty token list.")
        return None

    token_list_padded = pad_sequences([current_sequence_tokens], maxlen=max_model_input_len, padding='pre')

    if token_list_padded.shape[1] == 0 and len(current_sequence_tokens) > 0:
        st.error(f"Model input sequence length became 0 after padding. Max model input len: {max_model_input_len}")
        return None # Error condition

    predicted_probs = model_loaded.predict(token_list_padded, verbose=0)[0]
    predicted_probs = np.asarray(predicted_probs).astype('float64')
    epsilon = 1e-8
    predicted_probs = np.log(predicted_probs + epsilon) / temperature
    exp_preds = np.exp(predicted_probs)
    predicted_probs = exp_preds / np.sum(exp_preds)

    try:
        predicted_id = np.random.choice(len(predicted_probs), p=predicted_probs)
    except ValueError as e:
        if "sum(p)" in str(e) or "probabilities do not sum to 1" in str(e):
            st.warning(f"Probability sum issue: {e}. Using argmax for this step.")
            predicted_id = np.argmax(predicted_probs)
        else:
            raise e

    for word, index in tokenizer_loaded.word_index.items():
        if index == predicted_id:
            return word
    return None # Should not happen if predicted_id is valid

def generate_text(model_loaded, tokenizer_loaded, seed_text, num_words_to_generate, max_sequence_len_for_padding, temperature=0.7):
    """
    Generates text using the loaded model.
    max_sequence_len_for_padding: The original sequence length used for padding (model_input_len + 1)
    """
    if not model_loaded or not tokenizer_loaded:
        st.error("Model or tokenizer not loaded. Cannot generate text.")
        return seed_text + " [Error: Model/Tokenizer missing]"
    
    if max_sequence_len_for_padding <= 1:
        st.error(f"max_sequence_len_for_padding ({max_sequence_len_for_padding}) is too small. Cannot determine model input length.")
        return seed_text + " [Error: Invalid max_sequence_len_for_padding]"

    max_model_input_len = max_sequence_len_for_padding - 1 # This is the actual input_length for the model

    original_seed_text = seed_text
    current_seed_text_for_tokens = seed_text.lower() # Use lowercase for tokenization consistency
    generated_poem_part = ""

    for _ in range(num_words_to_generate):
        # Tokenize the current seed text. Important: This should be consistent with training.
        # The tokenizer was fit on lemmatized text, but sequences were made from original (cleaned) text.
        # For generation, we feed the generated text back, so it should be tokenized directly.
        current_tokens = tokenizer_loaded.texts_to_sequences([current_seed_text_for_tokens])[0]

        if not current_tokens:
            st.warning(f"Current seed text '{current_seed_text_for_tokens}' resulted in no known tokens. Stopping generation.")
            break
        
        # The model was trained on sequences of length `max_model_input_len`.
        # We need to feed it the last `max_model_input_len` tokens if current_tokens is longer.
        if len(current_tokens) > max_model_input_len:
            input_tokens_for_model = current_tokens[-max_model_input_len:]
        else:
            input_tokens_for_model = current_tokens

        output_word = generate_single_word(model_loaded, tokenizer_loaded, input_tokens_for_model, max_model_input_len, temperature)

        if not output_word or output_word == "<unk>":
            # Optionally, try to predict again, stop, or simply skip adding <unk>
            continue # Skip adding <unk> or if no word was found

        generated_poem_part += " " + output_word
        # Append the new word (in its original case if we had it, or lowercase) to the seed for the next iteration
        current_seed_text_for_tokens += " " + output_word

    return original_seed_text + generated_poem_part

if __name__ == '__main__':
    # This part is for basic independent testing of the text_generator
    # You would need to mock or load a model and tokenizer here.
    print("Text generator module. For independent testing, mock or load model/tokenizer.")
    # Example (requires actual model/tokenizer files and model_loader.py to be runnable):
    # from model_loader import load_model_and_tokenizer
    # test_model, test_tokenizer, test_max_len = load_model_and_tokenizer()
    # if test_model and test_tokenizer and test_max_len > 0:
    #     test_seed = "The wind"
    #     print(f"Testing generation with seed: '{test_seed}'")
    #     generated = generate_text(test_model, test_tokenizer, test_seed, 10, test_max_len, 0.7)
    #     print(f"Generated: {generated}")
    # else:
    #     print("Could not load model/tokenizer for generator test.")
