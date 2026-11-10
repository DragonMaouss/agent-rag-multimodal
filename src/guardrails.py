import re
import logging
from typing import List, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel, ValidationError
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class GuardrailSecurityError(Exception):
    """Exception levée en cas de violation des règles de sécurité."""
    pass


class IntentType(str, Enum):
    INTERNAL_DOCS = "internal_docs"
    WEB_SEARCH = "web_search"
    CALCULATION = "calculation"
    GENERAL_CHAT = "general_chat"



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
                logger.warning("Injection directe détectée -> %s", pattern)
                raise GuardrailSecurityError("Requête bloquée : Tentative de jailbreak / injection directe détectée.")

        # 2. Anonymisation PII entrée
        sanitized = query
        detected_pii = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, sanitized):
                detected_pii.append(pii_type)
                sanitized = re.sub(pattern, f"[REDACTED_{pii_type}]", sanitized)

        if detected_pii:
            logger.info("PII Anonymisées : %s", detected_pii)

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
                logger.warning("Injection indirecte neutralisée dans le document %s", doc.metadata.get("source"))
                continue
            safe_docs.append(doc)
        return safe_docs


class CalculatorArgsSchema(BaseModel):
    expression: str

class WebSearchArgsSchema(BaseModel):
    query: str

class InternalDocSearchArgsSchema(BaseModel):
    query: str


class ToolGuardrails:
    """Autorisations par rôle, validation d'arguments et contrôle d'action."""

    # Matrice de permissions d'outils par rôle utilisateur
    TOOL_PERMISSIONS = {
        "search_internal_documents": ["public", "tech", "rh", "admin"],
        "calculator_tool": ["public", "tech", "rh", "admin"],
        "web_search_tool": ["tech", "rh", "admin"],  # Restreint aux membres connectés (non guest)
        "fetch_web_page_tool": ["tech", "admin"]
    }

    # Schemas de validation d'arguments
    TOOL_SCHEMAS = {
        "calculator_tool": CalculatorArgsSchema,
        "web_search_tool": WebSearchArgsSchema,
        "search_internal_documents": InternalDocSearchArgsSchema
    }

    def validate_tool_call(self, tool_name: str, tool_args: Dict[str, Any], user_roles: List[str]) -> Dict[str, Any]:
        # 1. Contrôle des autorisations (RBAC Outils)
        allowed_roles = self.TOOL_PERMISSIONS.get(tool_name, ["admin"])
        if not any(role in allowed_roles for role in user_roles):
            logger.warning("Rôle insuffisant pour l'outil %s (Rôles: %s)", tool_name, user_roles)
            raise GuardrailSecurityError(f"Permission refusée : Vous n'avez pas l'autorisation d'utiliser l'outil '{tool_name}'.")

        # 2. Validation stricts des arguments (Anti-injection dans les arguments d'outils)
        schema = self.TOOL_SCHEMAS.get(tool_name)
        if schema:
            try:
                validated_args = schema(**tool_args).model_dump()
            except ValidationError as e:
                logger.warning("Arguments invalides pour %s -> %s", tool_name, e)
                raise GuardrailSecurityError(f"Arguments d'outil invalides pour {tool_name}.")
        else:
            validated_args = tool_args

        # 3. Contrôle d'injection dans l'expression de calcul
        if tool_name == "calculator_tool":
            expr = validated_args.get("expression", "")
            if any(char in expr for char in ["import", "exec", "eval", "__", "open"]):
                raise GuardrailSecurityError("Tentative d'injection de code détectée dans la calculatrice.")

        return validated_args


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
                logger.error("Tentative de fuite de jeton/clé détectée dans la réponse!")
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


class FinalResponseGuardrails:
    """Formatage final scellé avec enveloppe d'audit et métadonnées de sécurité."""

    def seal_response(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        detected_pii: List[str],
        intent: str,
        cache_hit: bool = False
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "tool_calls": tool_calls,
            "cache_hit": cache_hit,
            "security_envelope": {
                "status": "SECURE",
                "intent_classified": intent,
                "detected_pii": detected_pii,
                "guardrails_applied": ["InputValidation", "QueryProcessing", "RetrievalACL", "ToolAuth", "OutputSanitization"]
            }
        }