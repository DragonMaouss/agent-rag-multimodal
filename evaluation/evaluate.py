import os
import json
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# Chargement des variables d'environnement
load_dotenv()

API_URL = "http://127.0.0.1:8000/query"
DATASET_PATH = "evaluation/golden_dataset.json"

def check_keywords(response: str, expected_keywords: list, threshold: float) :
    """
    Verifie si la reponse passe le threshold de mot cles presents.
    """
    if not expected_keywords :
        return True

    matched = sum(1 for kw in expected_keywords if kw.lower() in response)

    ratio = matched/len(expected_keywords)
    return ratio >= threshold


def run_evaluation() :
    """"
    Evaluation sur "golden_dataset.json" avec un filtre par mot cles et une note par un LLM (1 a 5).
    """
    if not os.path.exists(DATASET_PATH):
        print(f"Erreur : le fichier {DATASET_PATH} est introuvable.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("--- Lancement de l'évalutation ---")

    judge_llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

    total_score = 0

    for item in dataset:
        test_id = item["id"]
        question = item["question"]
        expected = item["ground_truth"]
        keywords = item.get("expected_keywords", [])

        print(f"\nTest de la question {test_id} : {question}")

        # Appel de l'API RAG
        try:
            response = requests.post(API_URL, json={"query" : question})
            if response.status_code != 200 :
                print(f"Erreur API : ({response.status_code : {response.text}})")
                continue

            answer = response.json().get("answer", "")
            print(f"Réponse de l'agent : {answer}")
        except Exception as e:
            print(f"Impossible de contacter l'API : {e}")
            continue

        # Vérification par mots clés
        keywords_matched = check_keywords(answer.lower(), keywords, 0.5)
        
        # Evalutation par LLM
        judge_prompt =(
            f"Tu es un évaluateur expert. Note la réponse de l'agent par rapport à la réponse de référence sur une échelle de 1 à 5.\n\n"
            f"Question : {question}\n"
            f"Réponse de référence (Ground Truth) : {expected}\n"
            f"Réponse de l'agent : {answer}\n\n"
            f"Réponds UNIQUEMENT par un chiffre entier de 1 à 5 représentant la note."
        )

        try: 
            judge_res = judge_llm.invoke([HumanMessage(content=judge_prompt)])
            score = int(judge_res.content[0]["text"]) if judge_res else 0
        except Exception:
            score = 0

        print(f"Mots clés validés : {"Oui" if keywords_matched else "Non"}")
        print(f"Note du LLM juge : {score}/5")

        if score >= 4 and keywords_matched :
            total_score += 1

    succes_rate = (total_score / len(dataset)) * 100
    print(f"\n--- Résultat Final ---")

    print(f"Taux de réussite global : {succes_rate:.1f}% ({total_score}/{len(dataset)}) tests validés")
            
if __name__ == "__main__" :
    run_evaluation()


                        