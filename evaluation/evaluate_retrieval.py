import os
import json
import math
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


def calculate_map(relevance_label: list):
    """
    Calculate l'average precision (AP)
    """
    if not relevance_label or sum(relevance_label) == 0:
        return 0.0

    score = 0.0
    num_relevant_found = 0

    for i, label in enumerate(relevance_label, start=1):
        if label == 1:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / i
            score += precision_at_i

    total_relevant = sum(relevance_label)
    return score / total_relevant if total_relevant > 0 else 0.0


def calculate_ndcg(relevance_labels: list, k: int):
    """
    Calcule le Normalized Discounted Cumulative Gaint (NDCG@K)
    """
    if not relevance_labels or sum(relevance_labels) == 0 or k == 0:
        return 0.0

    # Calcul DCG
    dcg = 0.0
    for i, label in enumerate(relevance_labels[:k], start=1):
        if label == 1:
            dcg += 1.0 / math.log2(i+1)

    # Calcul IDCG
    ideal_labels = sorted(relevance_labels, reverse=True)
    idcg = 0.0
    for i, label in enumerate(ideal_labels[:k], start=1):
        if label == 1:
            idcg += 1.0 / math.log2(i+1)

    return (dcg / idcg) if idcg > 0 else 0.0


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
    average_precisions = []
    ndcg_scores = []

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
                average_precisions.append(1.0)
                ndcg_scores.append(1.0)
                print(f"Test {test_id} (Negatif) - Succes (Aucun bruit recupere)")
            else:
                reciprocal_ranks.append(0.0)
                average_precisions.append(0.0)
                ndcg_scores.append(0.0)
                print(f"Test {test_id} (Negatif) - Echec (Bruit recupere)")
            continue

        relevance_labels = evaluate_relevance_per_doc(retrieved_docs,expected_sources)

        # Test positifs
        if any(relevance_labels):
            hits_at_k += 1
            best_rank = relevance_labels.index(1) + 1
            reciprocal_ranks.append(1.0 / best_rank)

            ap = calculate_map(relevance_labels)
            ndcg = calculate_ndcg(relevance_labels, top_k)
            average_precisions.append(ap)
            ndcg_scores.append(ndcg)

            found_ranks_display = [i for i, lbl in enumerate(relevance_labels, 1) if lbl == 1]
            print(f"Test {test_id} - Succès (Trouvé aux rangs {found_ranks_display})")
        else:
            reciprocal_ranks.append(0.0)
            average_precisions.append(0.0)
            ndcg_scores.append(0.0)
            print(f"Test {test_id} - Echec")

        # Calcul des metriques
        hit_rate = (hits_at_k / total_tests) * 100
        mrr = sum(reciprocal_ranks) / total_tests
        map_score = sum(average_precisions) / total_tests
        ndcg_mean = sum(ndcg_scores) / total_tests


    print("\n--- Résultats de l'Évaluation du Retrieval ---")
    print(f"Taux de succès (Hit Rate @ {top_k}) : {hit_rate:.1f}% ({hits_at_k}/{total_tests})")
    print(f"Mean Reciprocal Rank (MRR)    : {mrr:.3f}")
    print(f"Mean Average Precision (MAP)  : {map_score:.3f}")
    print(f"NDCG Moyen @{top_k}           : {ndcg_mean:.3f}")

if __name__ == "__main__":
    run_retrieval_evaluation()