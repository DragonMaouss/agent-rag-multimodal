import os
import json
from dotenv import load_dotenv
from src.rag_multimodal import get_retriever

# Chargement des variables d'environnement
load_dotenv()

DATASET_PATH = "evaluation/golden_dataset.json"


def evaluate_relevance_per_doc(retrieved_docs, expected_sources: list) :
    """
    Retourne une liste binaire (1 si le document est pertinant, 0 sinon).
    """
    if not expected_sources:
        return [0] * len(retrieved_docs)

    relevance_labels = []
    for doc in retrieved_docs:
        content = doc.page_content.lower()
        source_matched = any(src.lower()in content for src in expected_sources)
        relevance_labels.append(1 if source_matched else 0)
    return relevance_labels


def run_retrieval_evaluation(top_k: int = 5) :
    if not os.path.exists(DATASET_PATH):
        print(f"Erreur : le fichier {DATASET_PATH} est introuvable.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)


    retriever = get_retriever()

    hits_at_k = 0
    total_tests = len(dataset)
    reciprocal_ranks = []

    print(f"--- Lancement de l'évalutation du retrieval (K={top_k}) ---")

    for item in dataset : 
        test_id = item["id"]
        question = item["question"]
        expected_sources = item.get("expected_sources", [])

        try :
            retrieved_docs = retriever.invoke(question)[:top_k]
        except Exception as e:
            print(f"Erreur lors du retrieval pour la question {test_id} : {e}")
            retrieved_docs = []


        # Gestion des tests negatifs 
        if not expected_sources:
            if len(retrieved_docs) == 0:
                hits_at_k += 1
                reciprocal_ranks.append(1.0)
                print(f"Test {test_id} (Negatif) - Succes (Aucun bruit recupere)")
            else:
                reciprocal_ranks.append(0.0)
                print(f"Test {test_id} (Negatif) - Echec (Bruit recupere)")
            continue

        relevance_labels = evaluate_relevance_per_doc(retrieved_docs,expected_sources)

        # Test positifs
        if any(relevance_labels):
            hits_at_k += 1
            best_rank = relevance_labels.index(1) + 1
            reciprocal_ranks.append(1.0 / best_rank)
            found_ranks_display = [i for i, lbl in enumerate(relevance_labels, 1) if lbl == 1]
            print(f"Test {test_id} - Succès (Trouvé aux rangs {found_ranks_display})")
        else:
            reciprocal_ranks.append(0.0)
            print(f"Test {test_id} - Echec")

        # Calcul des metriques
        hit_rate = (hits_at_k / total_tests) * 100
        mrr = sum(reciprocal_ranks) / total_tests


    print("\n--- Résultats de l'Évaluation du Retrieval ---")
    print(f"Taux de succès (Hit Rate @ {top_k}) : {hit_rate:.1f}% ({hits_at_k}/{total_tests})")
    print(f"Mean Reciprocal Rank (MRR)    : {mrr:.3f}")

if __name__ == "__main__":
    run_retrieval_evaluation()