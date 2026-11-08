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
st.markdown("Posez vos questions sur vos documents multiformats (PDF, Word, Excel, PowerPoint, Markdown, HTML).")

# URL de l'API FastAPI
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# Initialisation de l'historique des messages dans la session Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input pour poser une question ou télécharger un document
prompt = st.chat_input(
    "Posez votre question ou téléchargez un document...",
    accept_file=True,
    file_type=["pdf", "docx", "pptx", "xlsx", "html", "md"]
)

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

            with st.spinner(f"Indexation de {uploaded_file.name} ..."):
                mime_type = uploaded_file.type or "application/octet-stream"
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime_type)}
                try : 
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200 :
                        success_msg = f"{uploaded_file.name} indexé avec succès."
                        st.success(success_msg)
                        st.session_state.messages.append({"role": "assistant", "content": success_msg})
                    else :
                        error_msg = f"Erreur d'indexation ({res.status_code}) : {res.text}"
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
            with st.spinner("L'agent RAG réfléchit..."):
                try : 
                    payload = {
                        "query" : user_text,
                        "chat_history" : st.session_state.messages[-5: -1] if len(st.session_state.messages) > 1 else [] 
                    }
                    reponse = requests.post(f"{API_URL}/query", json=payload)
                    if reponse.status_code == 200:
                        data = reponse.json()
                        answer = data.get("answer", "Aucune réponse trouvée.")
                        sources = data.get("sources", [])
                        tool_calls = data.get("tool_calls", [])

                        st.markdown(answer)

                        # Affichage des outils MCP/Tools utilisés
                        if tool_calls:
                            with st.expander("🛠️ Outils & MCP exécutés"):
                                for tc in tool_calls:
                                    st.markdown(f"- **{tc.get('name')}** avec arguments : `{tc.get('args')}`")
                    else:
                        answer = f"Erreur API ({reponse.status_code})"
                        sources = []
                except Exception as e:
                    answer = f"Une erreur est survenue : {str(e)}"
                    sources = []

                st.markdown(answer)

                # Affichage des sources
                if sources :
                    with st.expander("Sources consultées") :
                        for src in sources:
                            s_name = src.get("source")
                            p_num = src.get("page")
                            img_i = src.get("image_index")
                            if img_i is not None:
                                st.markdown(f"- **{s_name}** (Page {p_num} — Image/Schéma #{img_i})")
                            else:
                                st.markdown(f"- **{s_name}** (Page {p_num})")

                assistant_full = answer
                if sources:
                    assistant_full += "\n\n*Sources :* " + ", ".join([f"{s['source']} (p. {s['page']})" for s in sources])

                st.session_state.messages.append({"role": "assistant", "content": assistant_full})