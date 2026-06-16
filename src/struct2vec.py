# src/struct2vec.py

import os
from collections import Counter

import numpy as np

_EXTENSIONS = [
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".cc",
    ".cs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh", ".fish",
    ".yaml", ".yml", ".json", ".xml", ".toml", ".cfg", ".ini",
    ".html", ".css", ".scss", ".less",
    ".md", ".rst", ".txt",
    ".sql", ".proto", ".tf",
]


class Struct2Vec:
    def __init__(self, repo_path, vector_size):
        self.repo_path = repo_path
        self.vector_size = vector_size

    def generate(self):
        ext_counts: Counter = Counter()
        dir_depths: list[int] = []
        file_count = 0
        dir_count = 0

        base_depth = self.repo_path.rstrip("/").count("/")

        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules")]
            depth = root.count("/") - base_depth
            dir_depths.append(depth)
            dir_count += len(dirs)
            file_count += len(files)
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext:
                    ext_counts[ext] += 1

        total = max(file_count, 1)
        max_depth = max(dir_depths) if dir_depths else 0
        mean_depth = sum(dir_depths) / len(dir_depths) if dir_depths else 0.0

        # Extension frequencies (one slot per known extension)
        features: list[float] = [ext_counts.get(ext, 0) / total for ext in _EXTENSIONS]

        # Global structural stats, soft-capped to [0, 1]
        features += [
            min(file_count / 500, 1.0),
            min(dir_count / 100, 1.0),
            min(max_depth / 10, 1.0),
            min(mean_depth / 5, 1.0),
        ]

        # Pad or truncate to vector_size
        arr = np.array(features, dtype=np.float32)
        if len(arr) < self.vector_size:
            arr = np.pad(arr, (0, self.vector_size - len(arr)))
        else:
            arr = arr[: self.vector_size]

        return arr
