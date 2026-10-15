import os
import json
from dotenv import load_dotenv
from src.rag_multimodal import get_retriever

# Chargement des variables d'environnement
load_dotenv()

DATASET_PATH = "evaluation/golden_dataset.json"


def check_retrieved_keywords(retrieved_docs, expected_keywords: list, threshold: float):
    if not expected_keywords:
        return True

    combined_context = " ".join([doc.page_content.lower() for doc in retrieved_docs])

    matched = sum(1 for kw in expected_keywords if kw.lower() in combined_context)
    
    ratio = matched/len(expected_keywords)
    return ratio >= threshold


def run_retrieval_evaluation() :
    if not os.path.exists(DATASET_PATH):
        print(f"Erreur : le fichier {DATASET_PATH} est introuvable.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)


    retriever = get_retriever()

    hits_at_k = 0
    total_tests = len(dataset)

    print("--- Lancement de l'évalutation du retrieval ---")

    for item in dataset : 
        test_id = item["id"]
        question = item["question"]
        excepted_keywords = item.get("expected_keywords", [])

        try :
            retrieved_docs = retriever.invoke(question)
        except Exception as e:
            print(f"Erreur lors du retrieval pour la question {test_id} : {e}")

        success = check_retrieved_keywords(retrieved_docs, excepted_keywords, threshold=0.5)

        if success:
            hits_at_k += 1
            
        print(f"Test {test_id} - Contexte pertinent trouvé : {'Oui' if success else 'Non'}")

    hit_rate = (hits_at_k / total_tests) * 100

    print("\n--- Résultats de l'Évaluation du Retrieval ---")
    print(f"Taux de succès du Retrieval (Context Hit Rate) : {hit_rate:.1f}% ({hits_at_k}/{total_tests})")

if __name__ == "__main__":
    run_retrieval_evaluation()