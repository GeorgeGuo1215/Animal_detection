"""
Giant Panda RAG — Retrieval-Augmented Generation for panda knowledge base.

Standalone module (adapted from agentAndRag/RAG/simple_rag).
Processes .txt files extracted from PDFs.
"""

from .query_rewrite import NoRewrite, TemplateRewriter
from .retrieval import MultiRouteRetriever, build_default_multiroute
