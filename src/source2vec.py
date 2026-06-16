# src/source2vec.py

import os
from itertools import islice

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


EXTENSION_TO_LANGUAGE = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.java': 'java',
    '.go': 'go',
    '.rs': 'rust',
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.hpp': 'cpp',
    '.cs': 'c_sharp',
    '.rb': 'ruby',
}

FUNCTION_NODE_TYPES = {
    'python': {'function_definition'},
    'javascript': {'function_declaration', 'method_definition', 'arrow_function'},
    'typescript': {'function_declaration', 'method_definition', 'arrow_function'},
    'tsx': {'function_declaration', 'method_definition', 'arrow_function'},
    'java': {'method_declaration', 'constructor_declaration'},
    'go': {'function_declaration', 'method_declaration'},
    'rust': {'function_item'},
    'c': {'function_definition'},
    'cpp': {'function_definition'},
    'c_sharp': {'method_declaration', 'constructor_declaration'},
    'ruby': {'method', 'singleton_method'},
}


def _load_ts_language(lang_name):
    """Return a tree-sitter Language for lang_name, or None if not installed."""
    specs = {
        "python": ("tree_sitter_python", "language"),
        "javascript": ("tree_sitter_javascript", "language"),
        "typescript": ("tree_sitter_typescript", "language_typescript"),
        "tsx": ("tree_sitter_typescript", "language_tsx"),
        "java": ("tree_sitter_java", "language"),
        "go": ("tree_sitter_go", "language"),
        "rust": ("tree_sitter_rust", "language"),
        "c": ("tree_sitter_c", "language"),
        "cpp": ("tree_sitter_cpp", "language"),
        "c_sharp": ("tree_sitter_c_sharp", "language"),
        "ruby": ("tree_sitter_ruby", "language"),
    }

    aliases = {
        "js": "javascript",
        "ts": "typescript",
        "csharp": "c_sharp",
        "cs": "c_sharp",
        "c++": "cpp",
    }

    lang_name = aliases.get(lang_name, lang_name)
    spec = specs.get(lang_name)
    if spec is None:
        return None

    module_name, fn_name = spec

    try:
        module = __import__(module_name)
        language = getattr(module, fn_name)()
        from tree_sitter import Language

        return language if isinstance(language, Language) else Language(language)
    except (ImportError, AttributeError, TypeError):
        return None


class Source2Vec:
    """Source code embeddings via tree-sitter AST parsing and CodeBERT.

    Supports any language with a tree-sitter grammar installed. Unrecognised
    file types are silently skipped, so a mixed-language repo works out of
    the box as long as the relevant tree-sitter-* packages are installed.

    CodeBERT outputs 768-dim vectors. If vector_size differs, the result is
    truncated or zero-padded. For lossless embeddings set vector_size=768.
    """

    DEFAULT_MODEL = "microsoft/codebert-base"

    DEFAULT_MAX_SNIPPETS = 64
    DEFAULT_BATCH_SIZE = 16

    def __init__(self, repo_path, vector_size, model_name=None, max_snippets=None, batch_size=None):
        self.repo_path = repo_path
        self.vector_size = vector_size
        self.max_snippets = self.DEFAULT_MAX_SNIPPETS if max_snippets is None else max_snippets
        self.batch_size = self.DEFAULT_BATCH_SIZE if batch_size is None else batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = model_name or self.DEFAULT_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._parser_cache = {}

    def generate(self):
        snippets = self._extract_functions()
        if not snippets:
            return np.zeros(self.vector_size)
        snippets = self._sample_snippets(snippets)
        embeddings = [
            embedding
            for batch in self._batched(snippets, self.batch_size)
            for embedding in self._embed_batch(batch)
        ]
        return self._resize(np.mean(embeddings, axis=0))

    def _sample_snippets(self, snippets):
        if self.max_snippets <= 0 or len(snippets) <= self.max_snippets:
            return snippets
        if self.max_snippets == 1:
            return [snippets[len(snippets) // 2]]
        last = len(snippets) - 1
        return [
            snippets[round(i * last / (self.max_snippets - 1))]
            for i in range(self.max_snippets)
        ]

    def _batched(self, items, size):
        iterator = iter(items)
        while True:
            batch = list(islice(iterator, size))
            if not batch:
                break
            yield batch

    # ------------------------------------------------------------------
    # AST extraction
    # ------------------------------------------------------------------

    def _extract_functions(self):
        snippets = []
        for root, _, files in os.walk(self.repo_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                lang = EXTENSION_TO_LANGUAGE.get(ext)
                if lang is None:
                    continue
                parser = self._get_parser(lang)
                if parser is None:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'rb') as f:
                        source = f.read()
                    tree = parser.parse(source)
                    snippets.extend(self._collect_functions(tree.root_node, source, lang))
                except Exception:
                    pass
        return snippets

    def _collect_functions(self, root_node, source, lang):
        target_types = FUNCTION_NODE_TYPES.get(lang, set())
        results = []
        stack = [root_node]
        while stack:
            node = stack.pop()
            if node.type in target_types:
                text = source[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
                results.append(text)
            stack.extend(node.children)
        return results

    def _get_parser(self, lang):
        if lang not in self._parser_cache:
            try:
                from tree_sitter import Parser
                language = _load_ts_language(lang)
                if language is None:
                    self._parser_cache[lang] = None
                else:
                    parser = Parser()
                    if hasattr(parser, 'set_language'):
                        parser.set_language(language)
                    else:
                        parser.language = language
                    self._parser_cache[lang] = parser
            except Exception:
                self._parser_cache[lang] = None
        return self._parser_cache[lang]

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_batch(self, code_snippets):
        inputs = self.tokenizer(
            code_snippets,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def _resize(self, vector):
        n = len(vector)
        if n == self.vector_size:
            return vector
        if n > self.vector_size:
            return vector[:self.vector_size]
        return np.pad(vector, (0, self.vector_size - n))
