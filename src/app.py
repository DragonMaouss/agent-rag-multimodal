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

# Chat input acceptant les PDF
prompt = st.chat_input(
    "Posez votre question ici...",
    accept_file=True,
    file_type=["pdf"]
)

# Saisir une question
if prompt:
    user_text = prompt.text if hasattr(prompt, "text") else prompt.get("text","")
    uploaded_files = prompt.files if hasattr(prompt, "files") else prompt.get("files","")

    # 1.Traitement des fichiers attachées
    if uploaded_files : 
        for uploaded_file in uploaded_files : 
            
            file_indicator = f"Document joint : {uploaded_file.name}"
            st.session_state.messages.append({"role" : "user", "content" : file_indicator})
            with st.chat_message("user") : 
                st.markdown(file_indicator)

            with st.spinner(f"Extraction du texte, des images et indexation de {uploaded_file.name} ...") :
                files = {"file" : (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try : 
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200 :
                        success_msg = f"{uploaded_file.name} indexé avec succès."
                        st.success(success_msg)
                        st.session_state.messages.append({"role" : "assistant", "content" : success_msg })
                    else :
                        error_msg = f"Erreur lors de l'indexation : {res.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role" : "assistant", "content" : error_msg })

                except Exception as e : 
                    st.error(f"Impossible de contacter l'API : {e}") 

    # 2. Traitement de la question
    if user_text and user_text.strip() :
        # Ajouter le message de l'utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # Envoyer la question à l'API FastAPI
        with st.chat_message("assistant") :
            with st.spinner("L'agent RAG Multimodal réfléchit..."):
                try : 
                    reponse = requests.post(f"{API_URL}/query", json={"query": user_text})
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