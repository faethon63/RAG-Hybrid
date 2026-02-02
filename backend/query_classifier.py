"""
Query Classifier Module
Classifies queries into intent types to improve routing and response generation.
"""

import re
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of user query intents."""
    META_QUESTION = "meta"           # Questions ABOUT how to do something
    TASK_EXECUTION = "task"          # Requests to actually DO something
    INFORMATION = "info"             # General information requests
    TECHNICAL_CODING = "coding"      # Programming/library questions
    DOCUMENT_LOOKUP = "doc_lookup"   # Questions about specific documents
    CONVERSATION = "conversation"    # Greetings, thanks, corrections
    AUTOMATION = "automation"        # Complex multi-step automation requests


class QueryClassifier:
    """Classify user queries to improve routing and response generation."""

    # Meta-question patterns (questions about HOW to do something)
    META_PATTERNS = [
        r"what (?:do i|would i|should i|will i) need",
        r"what (?:do you|would you|can you) need",
        r"how (?:do i|can i|should i|would i)",
        r"what(?:'s| is) the (?:best|proper|correct) way",
        r"what tools? (?:do i|should i|would i|are)",
        r"what(?:'s| is) required",
        r"what steps",
        r"what are the requirements",
        r"how does .* work",
        r"what libraries?",
        r"which (?:library|package|module|tool)",
        r"recommend .* (?:library|package|tool)",
        r"what will (?:it|that) take",
        r"what implementation",
        r"what strategy",
    ]

    # Task execution patterns (requests to DO something)
    TASK_PATTERNS = [
        r"^(?:please |can you |could you )?(?:fill|complete|create|write|generate|make)",
        r"^(?:help me |assist me )?(?:fill|complete|write)",
        r"^do (?:this|it|the)",
        r"^(?:fill|complete) (?:this|the|my)",
        r"extract .* from",
        r"read .* and (?:fill|complete|summarize)",
        r"analyze (?:this|the|my)",
    ]

    # Technical/coding question patterns
    CODING_PATTERNS = [
        r"(?:python|javascript|java|rust|go|c\+\+) (?:library|package|module|code)",
        r"(?:pip|npm|cargo|maven) install",
        r"(?:PyPDF|pdfrw|reportlab|pypdf|pdf-lib)",
        r"programmat(?:ic|ically)",
        r"(?:api|sdk|library) for",
        r"code (?:to|for|that)",
        r"script (?:to|for|that)",
        r"automate .* with (?:python|code)",
        r"what .* (?:library|package|module)",
        r"(?:open source|oss) .* for",
    ]

    # Automation patterns (should route to deep_agent)
    AUTOMATION_PATTERNS = [
        r"automate",
        r"batch process",
        r"bulk (?:fill|process|update)",
        r"(?:multiple|all|every) (?:file|form|document)",
        r"workflow",
        r"(?:loop|iterate) (?:through|over)",
        r"programmatically (?:fill|read|write)",
        r"system (?:that will|to)",
        r"find .* information .* fill",
    ]

    # User correction patterns
    CORRECTION_PATTERNS = [
        r"^no[,!.]?\s",
        r"^not\s",
        r"^i (?:said|meant|want|am asking)",
        r"^(?:don't|do not|doesn't|does not)",
        r"^that's (?:not|wrong)",
        r"^(?:actually|instead)",
        r"^(?:ignore|skip|forget)",
        r"not .* needed",
        r"no .* (?:needed|required|necessary)",
        r"why (?:are you not|aren't you|don't you)",
        r"you (?:didn't|did not) understand",
        r"that's not what i",
    ]

    @classmethod
    def classify(cls, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Classify a query into intent type with metadata.

        Returns:
            Dict with:
                - intent: QueryIntent enum value
                - is_correction: bool - if user is correcting previous response
                - requires_documents: bool - if query references specific documents
                - is_technical: bool - if query is about programming/tools
                - confidence: float - classification confidence
                - matched_pattern: str - the pattern that matched (for debugging)
        """
        query_lower = query.lower().strip()

        result = {
            "intent": QueryIntent.INFORMATION,
            "is_correction": False,
            "requires_documents": False,
            "is_technical": False,
            "confidence": 0.5,
            "matched_pattern": None,
        }

        # Check for user corrections first (high priority)
        for pattern in cls.CORRECTION_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                result["is_correction"] = True
                result["confidence"] = 0.9
                result["matched_pattern"] = f"correction:{pattern}"
                logger.info(f"Query classified as CORRECTION: {pattern}")
                break

        # Check for technical/coding questions
        for pattern in cls.CODING_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                result["intent"] = QueryIntent.TECHNICAL_CODING
                result["is_technical"] = True
                result["confidence"] = 0.85
                result["matched_pattern"] = f"coding:{pattern}"
                logger.info(f"Query classified as TECHNICAL_CODING: {pattern}")
                return result

        # Check for automation requests (route to deep_agent)
        for pattern in cls.AUTOMATION_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                result["intent"] = QueryIntent.AUTOMATION
                result["confidence"] = 0.8
                result["matched_pattern"] = f"automation:{pattern}"
                logger.info(f"Query classified as AUTOMATION: {pattern}")
                return result

        # Check for meta-questions (about HOW to do something)
        for pattern in cls.META_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                result["intent"] = QueryIntent.META_QUESTION
                result["confidence"] = 0.8
                result["matched_pattern"] = f"meta:{pattern}"
                logger.info(f"Query classified as META_QUESTION: {pattern}")
                return result

        # Check for task execution requests
        for pattern in cls.TASK_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                result["intent"] = QueryIntent.TASK_EXECUTION
                result["requires_documents"] = True
                result["confidence"] = 0.8
                result["matched_pattern"] = f"task:{pattern}"
                logger.info(f"Query classified as TASK_EXECUTION: {pattern}")
                return result

        # Check for document references
        doc_refs = [
            r"(?:page|form|line|section)\s*\d+",
            r"in (?:the|my) (?:document|file|pdf)",
            r"according to",
            r"based on .* (?:document|file)",
        ]
        for pattern in doc_refs:
            if re.search(pattern, query_lower, re.IGNORECASE):
                result["requires_documents"] = True
                break

        logger.info(f"Query classified as {result['intent'].value} (default)")
        return result

    @classmethod
    def get_context_modifier(cls, classification: Dict[str, Any]) -> str:
        """
        Generate context modifier based on classification.
        This gets prepended to the system prompt to modify behavior.
        """
        modifiers = []

        if classification.get("is_correction"):
            modifiers.append(
                "IMPORTANT: The user is correcting or clarifying a previous response. "
                "Pay close attention to their correction and adjust accordingly. "
                "Do NOT repeat information they explicitly rejected. "
                "Acknowledge their correction and provide what they actually asked for."
            )

        intent = classification.get("intent")
        if intent == QueryIntent.META_QUESTION:
            modifiers.append(
                "RESPONSE MODE: META-QUESTION\n"
                "The user is asking ABOUT how to accomplish something, "
                "not requesting you to actually do it. Explain the process, tools, "
                "requirements, and implementation strategy. Use a conversational format. "
                "Do NOT use structured form-filling output format for this response."
            )

        if intent == QueryIntent.TECHNICAL_CODING:
            modifiers.append(
                "RESPONSE MODE: TECHNICAL/PROGRAMMING\n"
                "The user is asking a technical programming question. "
                "Focus on: Python libraries, APIs, code examples, installation commands. "
                "Do NOT suggest manual GUI tools like Adobe Acrobat unless specifically asked. "
                "Prefer open-source solutions (e.g., pdfrw, reportlab, PyPDF2 for PDFs)."
            )

        if intent == QueryIntent.AUTOMATION:
            modifiers.append(
                "RESPONSE MODE: AUTOMATION REQUEST\n"
                "The user wants to build an automated system. "
                "Focus on: architecture, tools needed, implementation steps, code structure. "
                "Explain what components are needed rather than doing the task manually."
            )

        if not classification.get("requires_documents"):
            modifiers.append(
                "CITATION CONSTRAINT: This query does not require referencing specific documents. "
                "Do NOT cite page numbers or document sections unless you have "
                "actually read and retrieved content from those documents using tools. "
                "Do NOT fabricate citations or instruction references."
            )

        return "\n\n".join(modifiers) if modifiers else ""


# Convenience function
def classify_query(query: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Convenience function to classify a query."""
    return QueryClassifier.classify(query, history)
