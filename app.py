
import streamlit as st
from model_loader import load_model_and_tokenizer
from text_generator import generate_text
from config import DEFAULT_TEMPERATURE, MODEL_PATH, TOKENIZER_PATH # For checking existence
import os

# --- Streamlit App UI ---
st.set_page_config(page_title="Poetry Generator", layout="wide")
st.title("📜 AI Poetry Companion 🖋️")
st.markdown("Trained on an English poetry corpus. Use the options below to generate or complete poems.")

# --- Load Model and Tokenizer ---
# This will be cached by st.cache_resource in model_loader
model, tokenizer, MAX_SEQ_LEN_FOR_PADDING = load_model_and_tokenizer()



if model and tokenizer and MAX_SEQ_LEN_FOR_PADDING > 0:
    st.sidebar.header("Generation Parameters")
    temperature = st.sidebar.slider(
        "Temperature (Creativity)",
        min_value=0.1,
        max_value=1.5,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
        help="Lower values make output more predictable; higher values make it more random."
    )

    tab1, tab2 = st.tabs(["✒️ Generate Short Poem", "✍️ Complete a Line"])

    with tab1:
        st.header("Generate a Short Poem")
        st.markdown("Provide a starting phrase (seed text) and let the AI continue it.")
        
        seed_input_poem = st.text_input(
            "Enter your seed phrase for the poem:",
            "The old house stood on a hill",
            key="seed_poem"
        )
        num_words_poem = st.slider(
            "Number of words to generate for the poem:",
            min_value=10,
            max_value=300,
            value=50,
            key="len_poem"
        )

        if st.button("Compose Poem", key="btn_poem"):
            if not seed_input_poem.strip():
                st.warning("Please enter a seed phrase to start the poem.")
            else:
                with st.spinner("AI is composing your poem... ✨"):
                    generated_content = generate_text(
                        model,
                        tokenizer,
                        seed_input_poem,
                        num_words_poem,
                        MAX_SEQ_LEN_FOR_PADDING, # Pass the full sequence length used for padding
                        temperature
                    )
                    st.subheader("Your AI-Generated Poem:")
                    st.markdown(f"> {generated_content}")

    with tab2:
        st.header("Complete a Line of Poetry")
        st.markdown("Enter an incomplete line of poetry, and the AI will try to complete it.")

        seed_input_line = st.text_input(
            "Enter the incomplete line:",
            "In fields of green, where shadows",
            key="seed_line"
        )
        num_words_line = st.slider(
            "Number of words to add for completion:",
            min_value=3,
            max_value=50,
            value=10,
            key="len_line"
        )

        if st.button("Complete Line", key="btn_line"):
            if not seed_input_line.strip():
                st.warning("Please enter an incomplete line.")
            else:
                with st.spinner("AI is completing your line... ✍️"):
                    completed_content = generate_text(
                        model,
                        tokenizer,
                        seed_input_line,
                        num_words_line,
                        MAX_SEQ_LEN_FOR_PADDING, # Pass the full sequence length used for padding
                        temperature
                    )
                    st.subheader("Completed Line:")
                    st.markdown(f"> {completed_content}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model Info**")
    st.sidebar.markdown(f"Input sequence length for model: `{MAX_SEQ_LEN_FOR_PADDING - 1 if MAX_SEQ_LEN_FOR_PADDING > 0 else 'N/A'}`")
    st.sidebar.markdown(f"Vocabulary size (tokenizer): `{len(tokenizer.word_index) if tokenizer else 'N/A'}`")

elif not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    st.error(
        f"Model or tokenizer files are missing. "
        f"Please ensure '{os.path.basename(MODEL_PATH)}' and '{os.path.basename(TOKENIZER_PATH)}' "
        f"are in the same directory as the application after training the model."
    )
    st.info("You need to run your training script first to generate these files.")
else:
    # This case means model/tokenizer loading failed for other reasons (detailed error in model_loader)
    st.error(
        "Failed to load the model or tokenizer. "
        "Please check the console/terminal for specific error messages that occurred during startup."
    )

if __name__ == '__main__':
    # The Streamlit app is run by 'streamlit run streamlit_app.py'
    # This block is not strictly necessary for Streamlit but can be a placeholder
    pass
