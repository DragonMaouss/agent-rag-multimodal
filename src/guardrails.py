import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class GuardrailSecurityError(Exception):
    """Exception levée en cas de violation des règles de sécurité."""
    pass


class InputGuardrails:
    """Validation de l'entrée : PII et détection d'injection directe."""
    
    PII_PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "PHONE_FR": r"(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    }
    
    DIRECT_INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above)\s+instructions",
        r"disregard\s+(all\s+)?prior\s+prompts",
        r"system\s+prompt\s+override",
        r"you\s+are\s+now\s+DAN",
        r"mode\s+développeur\s+activé",
        r"affiche\s+(tes|les)\s+consignes\s+secrètes",
        r"bypassing?\s+safety\s+filters"
    ]

    def process(self, query: str) -> Tuple[str, List[str]]:
        # 1. Détection d'injection directe
        lowered = query.lower()
        for pattern in self.DIRECT_INJECTION_PATTERNS:
            if re.search(pattern, lowered):
                logger.warning("Bloc 1 Triggered: Injection directe détectée -> %s", pattern)
                raise GuardrailSecurityError("Requête bloquée : Tentative de jailbreak / injection directe détectée.")

        # 2. Anonymisation PII entrée
        sanitized = query
        detected_pii = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, sanitized):
                detected_pii.append(pii_type)
                sanitized = re.sub(pattern, f"[REDACTED_{pii_type}]", sanitized)

        if detected_pii:
            logger.info("Bloc 1 PII Anonymisées : %s", detected_pii)

        return sanitized, detected_pii


class RetrievalGuardrails:
    """Filtrage des documents extraits et détection d'injection indirecte."""

    INDIRECT_INJECTION_PATTERNS = [
        r"\[system_note\]",
        r"<!--\s*override\s*-->",
        r"attention\s+ai\s*:\s*ignore",
        r"instructions\s+cachées\s*:"
    ]

    def validate_and_sanitize_docs(self, docs: List[Document]) -> List[Document]:
        safe_docs = []
        for doc in docs:
            content = doc.page_content
            # Détection d'injection indirecte dissimulée dans les documents
            is_malicious = any(re.search(p, content.lower()) for p in self.INDIRECT_INJECTION_PATTERNS)
            if is_malicious:
                logger.warning("Bloc 3 Triggered: Injection indirecte neutralisée dans le document %s", doc.metadata.get("source"))
                continue
            safe_docs.append(doc)
        return safe_docs


class OutputGuardrails:
    """Anonymisation PII sortie, vérification de fuite de tokens et des sources."""

    SENSITIVE_LEAKS = [
        r"sk-[a-zA-Z0-9]{32,}",  # Clés API
        r"AIzaSy[a-zA-Z0-9_-]{33}",
        r"eyJhbGciOiJIUzI1Ni"   # Jetons JWT
    ]

    def process(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        # 1. Anti-fuite de clés / tokens secrets
        for leak_pattern in self.SENSITIVE_LEAKS:
            if re.search(leak_pattern, answer):
                logger.error("Bloc 6 Triggered: Tentative de fuite de jeton/clé détectée dans la réponse!")
                return "Erreur de sécurité : La réponse générée a été bloquée car elle contenait des données sensibles."

        # 2. Anonymisation PII en sortie
        sanitized_answer = answer
        for pii_type, pattern in InputGuardrails.PII_PATTERNS.items():
            sanitized_answer = re.sub(pattern, f"[REDACTED_{pii_type}]", sanitized_answer)

        # 3. Validation des citations et des sources
        if sources and "Sources consultées" not in sanitized_answer and "*Sources :*" not in sanitized_answer:
            source_names = ", ".join(set(s.get("source", "Inconnu") for s in sources))
            sanitized_answer += f"\n\n*Sources vérifiées :* {source_names}"

        return sanitized_answer