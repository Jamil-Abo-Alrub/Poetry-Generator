import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
import re
import pickle
import os

# --- Constants (Adjust these as needed for your training) ---
VOCAB_SIZE = 10000      # Max number of words to keep in the vocabulary
EMBEDDING_DIM = 100     # Dimension of word embeddings
RNN_UNITS = 128         # Number of units in the LSTM layer
MAX_SEQUENCE_LEN = 50   # Maximum length of input sequences for training
EPOCHS = 50             # Number of epochs to train for
BATCH_SIZE = 64         # Batch size for training
CORPUS_FILE_PATH = "potxt.txt" # Path to your poetry text file
MODEL_SAVE_PATH = "poetry_generator_model.keras"
TOKENIZER_SAVE_PATH = "tokenizer.pickle"

# --- SpaCy Model Loading ---
def load_spacy_model(model_name="fr_core_web_sm"):
    """Loads or downloads a SpaCy model."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Downloading SpaCy model: {model_name}...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp

nlp = load_spacy_model()

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path, tokenizer_vocab_size, max_seq_len_param):
    """
    Loads poetry from a file, preprocesses it, tokenizes,
    and creates sequences for training an RNN.
    """
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: Corpus file not found at {file_path}")
        print("Please create this file with your poetry data, or provide the correct path.")
        print("Example content for poetry_corpus.txt:")
        print("the rain in spain falls mainly on the plain\n")
        print("a thing of beauty is a joy forever\n")
        return None, None, None, 0, None

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if not text.strip():
        print("Error: Corpus file is empty.")
        return None, None, None, 0, None

    # Basic cleaning
    text = text.lower()
    # Keep letters, basic punctuation relevant to poetry, and newlines
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ .',!\n]", " ", text)
    text = re.sub(r"\s+", " ", text).strip() # Normalize whitespace

    print("Lemmatizing text (this may take a while for large corpora)...")
    doc = nlp(text)
    
    lemmatized_tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    print(f"Number of lemmatized tokens: {len(lemmatized_tokens)}")

    if not lemmatized_tokens:
        print("Error: No lemmatized tokens found. Check corpus content and SpaCy processing.")
        return None, None, None, 0, None

    # Initialize and fit Keras Tokenizer on lemmatized text
    tokenizer = Tokenizer(num_words=tokenizer_vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts([" ".join(lemmatized_tokens)]) # Fit on the single string of lemmatized tokens
    print(f"Vocabulary size (from tokenizer): {len(tokenizer.word_index)}")

    
    corpus_lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not corpus_lines:
        print("Error: No lines found in the corpus after splitting by newline.")
        return None, None, None, 0, tokenizer

    input_sequences = []
    for line in corpus_lines:
        # Convert line to sequence of integers using the fitted tokenizer
        token_list = tokenizer.texts_to_sequences([line])[0]
        # Create n-gram sequences
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    if not input_sequences:
        print("Error: No input sequences generated. This might happen if lines are too short or all words are OOV.")
        return None, None, None, 0, tokenizer

    print(f"Number of raw sequences: {len(input_sequences)}")

    # Pad sequences to ensure uniform length
    # Determine the actual max sequence length from data, but cap it by max_seq_len_param
    actual_max_len_in_data = max([len(x) for x in input_sequences])
    current_max_len_for_padding = min(max_seq_len_param, actual_max_len_in_data)
    
    print(f"Max sequence length in data: {actual_max_len_in_data}")
    print(f"Using sequence length for padding: {current_max_len_for_padding}")


    padded_sequences = np.array(pad_sequences(input_sequences, maxlen=current_max_len_for_padding, padding='pre'))

    if padded_sequences.size == 0:
        print("Error: Padded sequences are empty. Check sequence generation and padding.")
        return None, None, None, 0, tokenizer

   
    X, labels_raw = padded_sequences[:,:-1], padded_sequences[:,-1]

    
    y = tf.keras.utils.to_categorical(labels_raw, num_classes=tokenizer_vocab_size)

    print(f"Shape of X (predictors): {X.shape}")
    print(f"Shape of y (labels): {y.shape}")

    return X, y, tokenizer, current_max_len_for_padding, lemmatized_tokens


# --- RNN Model Definition ---
def create_rnn_model(vocab_size_param, embedding_dim_param, rnn_units_param, sequence_len_param):
    """Creates and compiles the RNN model."""
    model = Sequential([
        # Embedding layer: Turns positive integers (indexes) into dense vectors of fixed size.
        # input_length is the length of input sequences, which is sequence_len_param - 1 (because the last word is the target).
        Embedding(input_dim=vocab_size_param,
                  output_dim=embedding_dim_param,
                  input_length=sequence_len_param-1),
        # LSTM layer: A type of recurrent layer.
        LSTM(rnn_units_param),
        # Dense layer: A regular densely-connected NN layer.
        # Activation 'softmax' makes the output represent a probability distribution over the vocabulary.
        Dense(vocab_size_param, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("\nModel Summary:")
    model.summary()
    return model

# --- Main Training Function ---
def train_and_save_model():
  
    print("--- Starting Poetry Model Training ---")

    # 1. Load and preprocess data
    X_train, y_train, tokenizer, processed_max_seq_len, _ = load_and_preprocess_data(
        CORPUS_FILE_PATH,
        VOCAB_SIZE,
        MAX_SEQUENCE_LEN
    )

    if X_train is None or y_train is None or tokenizer is None or X_train.shape[0] == 0:
        print("\nData loading or preprocessing failed. Exiting training.")
        return

    if X_train.shape[1] == 0:
        print("\nError: Predictor sequences (X_train) have zero length after processing.")
        print("This can happen if MAX_SEQUENCE_LEN is too small (e.g., 1 or 2) or all sequences were shorter.")
        print(f"Current processed_max_seq_len for padding was: {processed_max_seq_len}")
        print(f"X_train shape is {X_train.shape}. The second dimension should be > 0.")
        return

    print(f"\nUsing effective vocabulary size: {VOCAB_SIZE}")
    print(f"Using effective sequence length for model input: {processed_max_seq_len -1 }")


    # 2. Create the RNN model
    model = create_rnn_model(
        VOCAB_SIZE,         # Use the predefined VOCAB_SIZE for the model architecture
        EMBEDDING_DIM,
        RNN_UNITS,
        processed_max_seq_len # Use the actual sequence length determined from data (capped by MAX_SEQUENCE_LEN)
    )

    # 3. Train the model
    print(f"\n--- Training Model for {EPOCHS} epochs ---")
    try:
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        print("\n--- Model Training Completed ---")

        # 4. Save the trained model
        model.save(MODEL_SAVE_PATH)
        print(f"Model saved to: {MODEL_SAVE_PATH}")

        # 5. Save the tokenizer
        with open(TOKENIZER_SAVE_PATH, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer saved to: {TOKENIZER_SAVE_PATH}")

        print("\nTraining and saving process finished successfully!")
        print(f"You can now use '{MODEL_SAVE_PATH}' and '{TOKENIZER_SAVE_PATH}' in your Streamlit app.")

    except Exception as e:
        print(f"\nAn error occurred during training or saving: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # --- Create a dummy poetry_corpus.txt if it doesn't exist for testing ---
    if not os.path.exists(CORPUS_FILE_PATH):
        print(f"'{CORPUS_FILE_PATH}' not found. Creating a dummy file for demonstration.")
        print("Please replace this with your actual poetry corpus for meaningful training.")
      
    
    train_and_save_model()
