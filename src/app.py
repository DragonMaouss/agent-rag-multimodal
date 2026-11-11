import os
import streamlit as st
import requests

st.set_page_config(
    page_title="Agent RAG Multimodal",
    page_icon="🤖",
    layout="centered",
)

# URL de l'API FastAPI
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Initialisation des variables de session
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "roles" not in st.session_state:
    st.session_state.roles = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar pour l'authentification et le profil utilisateur
if not st.session_state.token:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("Connexion Agent RAG")
        st.markdown("Identifiez-vous pour accéder à l'assistant et à vos documents sécurisés.")

        with st.form("login_form", clear_on_submit=False):
            username_input = st.text_input("Nom d'utilisateur")
            password_input = st.text_input("Mot de passe", type="password")
            submit_button = st.form_submit_button("Se connecter", type="primary", use_container_width=True)

            if submit_button:
                if username_input and password_input:
                    try:
                        res = requests.post(
                            f"{API_URL}/login",
                            data={"username": username_input, "password": password_input},
                            headers={"Content-Type": "application/x-www-form-urlencoded"}
                        )
                        if res.status_code == 200:
                            data = res.json()
                            st.session_state.token = data.get("access_token")
                            st.session_state.username = data.get("username")
                            st.session_state.roles = data.get("roles", [])
                            st.success("Connexion réussie !")
                            st.rerun()
                        else:
                            st.error("Nom d'utilisateur ou mot de passe incorrect.")
                    except Exception as e:
                        st.error(f"Impossible de contacter l'API : {e}")
                else:
                    st.warning("Veuillez remplir tous les champs.")
else : 

    with st.sidebar:
        st.title("Profil & Session")
        st.success(f"Connecté : {st.session_state.username}")
        
        st.markdown("Privilèges / Rôles assignés :")
        for role in st.session_state.roles:
            st.markdown(f"- `{role}`")

        st.divider()

        # Rôles autorisés pour l'indexation
        st.subheader("Indexation de document")
        upload_roles_input = st.text_input(
            "Rôles autorisés pour le document",
            value="public",
            help="Séparés par des virgules (ex: public, tech, rh)"
        )

        st.divider()

        if st.button("Se déconnecter", type="secondary", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.roles = []
            st.session_state.messages = []
            st.rerun()

    # Page principale de l'application
    st.title("🤖 Agent RAG Multimodal")
    st.markdown("Posez vos questions sur vos documents multiformats (PDF, Word, Excel, PowerPoint, Markdown, HTML).")

    # Initialisation de l'historique des messages dans la session Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []


    if not st.session_state.token:
        st.info("Veuillez vous connecter pour poser des questions ou indexer des documents.")
    else :
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
            headers = {"Authorization": f"Bearer {st.session_state.token}"}

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
                        data = {"access_roles": upload_roles_input}
                        try : 
                            res = requests.post(f"{API_URL}/upload", files=files, headers=headers, json=data)
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
                            reponse = requests.post(f"{API_URL}/query", json=payload, headers=headers)
                            
                            if reponse.status_code == 200:
                                data = reponse.json()
                                answer = data.get("answer", "Aucune réponse trouvée.")
                                sources = data.get("sources", [])
                                tool_calls = data.get("tool_calls", [])
                                security_env = data.get("security_envelope", {})

                                st.markdown(answer)

                                # Métadonnées de l'enveloppe de sécurité
                                if security_env:
                                    with st.expander("🛡️ Enveloppe de Sécurité"):
                                        st.markdown(f"- Statut : `{security_env.get('status')}`")
                                        pii = security_env.get('detected_pii', [])
                                        if pii:
                                            st.markdown(f"- PII Anonymisées : `{', '.join(pii)}`")
                                        st.markdown(f"- Guardrails exécutés : `{', '.join(security_env.get('guardrails_applied', []))}`")                            

                                # Affichage des outils MCP/Tools utilisés
                                if tool_calls:
                                    with st.expander("Outils & MCP exécutés"):
                                        for tc in tool_calls:
                                            st.markdown(f"- {tc.get('name')} avec arguments : `{tc.get('args')}`")
                                
                                # Affichage des sources
                                if sources :
                                    with st.expander("Sources consultées") :
                                        for src in sources:
                                            s_name = src.get("source")
                                            p_num = src.get("page")
                                            img_i = src.get("image_index")
                                            if img_i is not None:
                                                st.markdown(f"- {s_name} (Page {p_num} — Image/Schéma #{img_i})")
                                            else:
                                                st.markdown(f"- {s_name} (Page {p_num})")                       
                            elif reponse.status_code == 400:
                                error_detail = reponse.json().get("detail", "Violation des règles de sécurité.")
                                answer = f"Guardrail Déclenché (400) : {error_detail}"
                                sources = []
                                st.warning(answer)
                            elif reponse.status_code in (401, 403):
                                error_detail = reponse.json().get("detail", "Accès non autorisé.")
                                answer = f"Sécurité ({reponse.status_code}) : {error_detail}"
                                sources = []
                                st.error(answer)
                            else:
                                answer = f"Erreur API ({reponse.status_code}) : {reponse.text}"
                                sources = []
                                st.error(answer)
                        except Exception as e:
                            answer = f"Une erreur est survenue : {str(e)}"
                            sources = []
                        
                        assistant_full = answer
                        if sources:
                            assistant_full += "\n\n*Sources :* " + ", ".join([f"{s['source']} (p. {s['page']})" for s in sources])

                        st.session_state.messages.append({"role": "assistant", "content": assistant_full})