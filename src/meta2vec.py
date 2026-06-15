# src/meta2vec.py

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class Meta2Vec:
    """Embeds repository metadata (README, description, topics) via mean-pooled BERT.

    Default model outputs 384 dims. Set vector_size=384 for lossless embeddings;
    smaller values truncate, larger values zero-pad.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, repo_path, vector_size, model_name=None):
        self.repo_path = repo_path
        self.vector_size = vector_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = model_name or self.DEFAULT_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(self):
        text = self._extract_metadata()
        if not text.strip():
            return np.zeros(self.vector_size)
        return self._resize(self._embed(text))

    def _extract_metadata(self):
        parts = [os.path.basename(self.repo_path)]
        for fname in ('DESCRIPTION', 'TOPICS'):
            fpath = os.path.join(self.repo_path, fname)
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    parts.append(f.read().strip())
        readme = self._get_readme()
        if readme:
            parts.append(readme)
        return ' '.join(parts)

    def _get_readme(self):
        for fname in ('README.md', 'README.txt', 'README'):
            fpath = os.path.join(self.repo_path, fname)
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        return None

    def _embed(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.squeeze().cpu().numpy()

    def _resize(self, vector):
        n = len(vector)
        if n == self.vector_size:
            return vector
        if n > self.vector_size:
            return vector[:self.vector_size]
        return np.pad(vector, (0, self.vector_size - n))
