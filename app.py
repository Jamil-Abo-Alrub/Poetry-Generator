import streamlit as st
from model_loader import load_model_and_tokenizer
from text_generator import generate_text
from config import DEFAULT_TEMPERATURE, MODEL_PATH, TOKENIZER_PATH # Pour vérifier l'existence
import os

# --- Interface Utilisateur de l'Application Streamlit ---
st.set_page_config(page_title="Générateur de Poésie", layout="wide") # MODIFIÉ
st.title("📜 Compagnon de Poésie IA 🖋️") # MODIFIÉ
# MODIFIÉ - Supposant que vous utiliserez un modèle entraîné sur un corpus français
st.markdown("Entraîné sur un corpus de poésie française. Utilisez les options ci-dessous pour générer ou compléter des poèmes.")


# --- Charger le Modèle et le Tokenizer ---
# Ceci sera mis en cache par st.cache_resource dans model_loader
model, tokenizer, MAX_SEQ_LEN_FOR_PADDING = load_model_and_tokenizer()



if model and tokenizer and MAX_SEQ_LEN_FOR_PADDING > 0:
    st.sidebar.header("Paramètres de Génération") # MODIFIÉ
    temperature = st.sidebar.slider(
        "Température (Créativité)", # MODIFIÉ
        min_value=0.1,
        max_value=1.5,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
        help="Les valeurs basses rendent le résultat plus prévisible ; les valeurs hautes le rendent plus aléatoire et créatif." # MODIFIÉ
    )

    # MODIFIÉ - Étiquettes des onglets en français
    tab1, tab2 = st.tabs(["✒️ Générer un Poème Court", "✍️ Compléter un Vers"])

    with tab1:
        st.header("Générer un Poème Court") # MODIFIÉ
        st.markdown("Fournissez une phrase de départ (texte d'amorce) et laissez l'IA la continuer.") # MODIFIÉ
        
        seed_input_poem = st.text_input(
            "Entrez votre phrase d'amorce pour le poème :", # MODIFIÉ
            "La vieille maison se dressait sur la colline", # MODIFIÉ - Exemple en français
            key="seed_poem"
        )
        num_words_poem = st.slider(
            "Nombre de mots à générer pour le poème :", # MODIFIÉ
            min_value=10,
            max_value=300,
            value=50,
            key="len_poem"
        )

        if st.button("Composer le Poème", key="btn_poem"): # MODIFIÉ
            if not seed_input_poem.strip():
                st.warning("Veuillez entrer une phrase d'amorce pour commencer le poème.") # MODIFIÉ
            else:
                with st.spinner("L'IA compose votre poème... ✨"): # MODIFIÉ
                    generated_content = generate_text(
                        model,
                        tokenizer,
                        seed_input_poem,
                        num_words_poem,
                        MAX_SEQ_LEN_FOR_PADDING, 
                        temperature
                    )
                    st.subheader("Votre Poème Généré par l'IA :") # MODIFIÉ
                    st.markdown(f"> {generated_content}")

    with tab2:
        st.header("Compléter un Vers de Poésie") # MODIFIÉ
        st.markdown("Entrez un vers de poésie incomplet, et l'IA essaiera de le compléter.") # MODIFIÉ

        seed_input_line = st.text_input(
            "Entrez le vers incomplet :", # MODIFIÉ
            "Dans les champs verts, où les ombres", # MODIFIÉ - Exemple en français
            key="seed_line"
        )
        num_words_line = st.slider(
            "Nombre de mots à ajouter pour la complétion :", # MODIFIÉ
            min_value=3,
            max_value=50,
            value=10,
            key="len_line"
        )

        if st.button("Compléter le Vers", key="btn_line"): # MODIFIÉ
            if not seed_input_line.strip():
                st.warning("Veuillez entrer un vers incomplet.") # MODIFIÉ
            else:
                with st.spinner("L'IA complète votre vers... ✍️"): # MODIFIÉ
                    completed_content = generate_text(
                        model,
                        tokenizer,
                        seed_input_line,
                        num_words_line,
                        MAX_SEQ_LEN_FOR_PADDING, 
                        temperature
                    )
                    st.subheader("Vers Complété :") # MODIFIÉ
                    st.markdown(f"> {completed_content}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Informations sur le Modèle**") # MODIFIÉ
    st.sidebar.markdown(f"Longueur de séquence d'entrée pour le modèle : `{MAX_SEQ_LEN_FOR_PADDING - 1 if MAX_SEQ_LEN_FOR_PADDING > 0 else 'N/A'}`") # MODIFIÉ
    st.sidebar.markdown(f"Taille du vocabulaire (tokenizer) : `{len(tokenizer.word_index) if tokenizer else 'N/A'}`") # MODIFIÉ

elif not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    st.error(
        f"Les fichiers du modèle ou du tokenizer sont manquants. " # MODIFIÉ
        f"Veuillez vous assurer que '{os.path.basename(MODEL_PATH)}' et '{os.path.basename(TOKENIZER_PATH)}' " 
        f"se trouvent dans le même répertoire que l'application après l'entraînement du modèle." # MODIFIÉ
    )
    st.info("Vous devez d'abord exécuter votre script d'entraînement pour générer ces fichiers.") # MODIFIÉ
else:
    st.error(
        "Échec du chargement du modèle ou du tokenizer. " # MODIFIÉ
        "Veuillez vérifier la console/le terminal pour les messages d'erreur spécifiques survenus au démarrage." # MODIFIÉ
    )

if __name__ == '__main__':
    pass