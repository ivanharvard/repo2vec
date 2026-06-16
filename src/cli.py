import json
import sys
import typer
from pathlib import Path
from repo2vec import Repo2Vec

app = typer.Typer(add_completion=False)

@app.command()
def embed(
    path: str = typer.Argument(..., help="Path to a local repository"),
    metadata: list[str] = typer.Option([], help="Extra metadata files (absolute paths or relative to repo)"),
    vector_size: int = typer.Option(128, help="Embedding dimension per component (meta, struct, source)"),
    combination_method: str = typer.Option("weighted_sum", help="concatenate | sum | average | median | weighted_sum"),
    normalize: bool = typer.Option(True, help="Normalize vectors before and after combining"),
):
    """Generate a JSON embedding for a local repository and print to stdout."""
    repo_path = Path(path).resolve()
    if not repo_path.is_dir():
        typer.echo(f"error: {path} is not a directory", err=True)
        raise typer.Exit(1)

    r2v = Repo2Vec(
        str(repo_path),
        vector_size=vector_size,
        combination_method=combination_method,
        normalize=normalize,
        metadata_files=metadata if metadata else None,
    )
    embedding = r2v.generate_embedding()

    print(json.dumps({"repo": str(repo_path), "embedding": embedding.tolist()}))


def main():
    app()
