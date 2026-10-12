import os
import streamlit as st
import requests

st.set_page_config(
    page_title="Agent RAG Multimodal",
    page_icon="🤖",
    layout="centered",
)

# Titre de l'application
st.title("🤖 Agent RAG Multimodal")
st.markdown("Posez vos questions sur le document multimodal et obtenez des réponses précises basées sur le texte et les images.")

# URL de l'API FastAPI
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# Initialisation de l'historique des messages dans la session Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Saisir une question
if prompt := st.chat_input("Posez votre question ici..."):
    # Ajouter le message de l'utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Envoyer la question à l'API FastAPI
    with st.chat_message("assistant") :
        with st.spinner("L'agent RAG Multimodal réfléchit..."):
            try : 
                reponse = requests.post(f"{API_URL}/query", json={"query": prompt})
                if reponse.status_code == 200:
                    answer = reponse.json().get("answer", "Aucune réponse trouvée.")
                else:
                    answer = f"Erreur lors de la requête à l'API : {reponse.status_code}"
            except requests.exceptions.ConnectionError as e:
                answer = f"Erreur de connexion à l'API : {str(e)}"
            except Exception as e:
                answer = f"Une erreur est survenue : {str(e)}"

            st.markdown(answer)
            # Ajouter la réponse de l'assistant à l'historique
            st.session_state.messages.append({"role": "assistant", "content": answer})