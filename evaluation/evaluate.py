import os
import json
import re
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.config import settings

# Chargement des variables d'environnement
load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000") + "/query"
DATASET_PATH = "evaluation/golden_dataset.json"


def check_keywords(response: str, expected_keywords: list, threshold: float = 0.4) -> bool:
    """
    Vérifie si le ratio de mots-clés présents dans la réponse atteint le seuil (threshold).
    """
    if not expected_keywords:
        return True

    matched = sum(1 for kw in expected_keywords if kw.lower() in response.lower())
    return (matched / len(expected_keywords)) >= threshold


def parse_llm_score(text: str) -> int:
    """
    Extrait proprement un entier de 1 à 5 depuis le texte retourné par le LLM juge.
    """
    match = re.search(r"\b([1-5])\b", text)
    if match:
        return int(match.group(1))
    return 0


def run_evaluation():
    if not os.path.exists(DATASET_PATH):
        print(f"Erreur : le fichier {DATASET_PATH} est introuvable.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("--- Lancement de l'évaluation globale des réponses ---")
    judge_llm = ChatGoogleGenerativeAI(model=settings.gemini_model)

    total_score = 0

    for item in dataset:
        test_id = item["id"]
        question = item["question"]
        expected = item["ground_truth"]
        keywords = item.get("expected_keywords", [])

        print(f"\nTest {test_id} : {question}")

        try:
            response = requests.post(API_URL, json={"query": question})
            if response.status_code != 200:
                print(f"Erreur API ({response.status_code}) : {response.text}")
                continue

            answer = response.json().get("answer", "")
            print(f"Réponse agent : {answer}")
        except Exception as e:
            print(f"Impossible de contacter l'API : {e}")
            continue

        keywords_matched = check_keywords(answer, keywords, threshold=0.5)

        judge_prompt = (
            "Tu es un évaluateur expert. Note la réponse de l'agent par rapport à la réponse de référence sur une échelle de 1 à 5.\n\n"
            f"Question : {question}\n"
            f"Réponse de référence : {expected}\n"
            f"Réponse de l'agent : {answer}\n\n"
            "Réponds uniquement par un chiffre entier entre 1 et 5."
        )

        try:
            judge_res = judge_llm.invoke([HumanMessage(content=judge_prompt)])
            score = parse_llm_score(str(judge_res.content))
        except Exception as e:
            print(f"Erreur évaluation LLM juge : {e}")
            score = 0

        print(f"Mots-clés validés : {'Oui' if keywords_matched else 'Non'}")
        print(f"Note du LLM juge : {score}/5")

        if score >= 4 and (keywords_matched or score == 5):
            total_score += 1

    success_rate = (total_score / len(dataset)) * 100
    print("\n--- Résultat Final ---")
    print(f"Taux de réussite global : {success_rate:.1f}% ({total_score}/{len(dataset)} tests validés)")


if __name__ == "__main__":
    run_evaluation()