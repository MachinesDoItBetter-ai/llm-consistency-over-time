#!/usr/bin/env python3

import argparse
import os
import re
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Robust similarity scoring
# ---------------------------------------------------------

def compute_similarity_scores(sim_matrix: np.ndarray) -> dict:
    """
    Compute several similarity scores for a set of documents.

    - mean:   mean of all off-diagonal similarities
    - median: median of all off-diagonal similarities
    - core:   mean of top half of similarities (robust to outliers)
    """
    n = sim_matrix.shape[0]
    if n < 2:
        return {"mean": float("nan"), "median": float("nan"), "core": float("nan")}

    mask = ~np.eye(n, dtype=bool)
    vals = sim_matrix[mask]

    mean_score = float(vals.mean())
    median_score = float(np.median(vals))

    sorted_vals = np.sort(vals)
    half = len(sorted_vals) // 2
    core_mean_score = float(sorted_vals[half:].mean())

    return {
        "mean": mean_score,
        "median": median_score,
        "core": core_mean_score,
    }

def tokenize(text: str, words_only: bool) -> list[str]:
    """
    Tokenizer:
    - lowercase
    - if words_only=True: strip all digits & punctuation, keep alphabetic words only
    - else: include alphanumeric tokens
    """
    text = text.lower()

    if words_only:
        # remove all digits and punctuation
        text = re.sub(r"[^a-z\s]", " ", text)
        # extract pure alphabetic words
        return re.findall(r"[a-z]+", text)

    return re.findall(r"\w+", text)

def longest_common_prefix(strings: list[str]) -> str:
    """
    Return the longest common prefix of a list of strings.
    If there is no common prefix, return an empty string.
    """
    if not strings:
        return ""
    if len(strings) == 1:
        return strings[0]

    # Lexicographic min / max trick
    s1 = min(strings)
    s2 = max(strings)
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return s1[:i]
# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main(args) -> None:
   
    # Filter file list
    file_paths = []
    for path in args.files:
        if os.path.isfile(path):
            file_paths.append(path)
        else:
            print(f"Warning: '{path}' is not a file, skipping.", file=sys.stderr)

    if len(file_paths) < 2:
        print("Need at least two valid text files to compare.", file=sys.stderr)
        sys.exit(1)

    # Read documents
    texts = []
    labels = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        if not content:
            print(f"Warning: '{path}' is empty, skipping.", file=sys.stderr)
            continue
        texts.append(content)
        # labels.append(os.path.basename(path))

    if len(texts) < 2:
        print("Not enough non-empty files to compare.", file=sys.stderr)
        sys.exit(1)

    stems = [
        os.path.splitext(os.path.basename(p))[0]
        for p in file_paths
    ]
    prefix = longest_common_prefix(stems)
    if not prefix:
        prefix = "output"
    prefix = re.sub(r"\d+", "", prefix)
    prefix = re.sub(r"[^\w\-]", "", prefix)
    prefix = re.sub(r"-{2,}", "-", prefix)
    prefix = prefix.strip("-_")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_prefix = os.path.join(results_dir, prefix)

    # TF-IDF + cosine similarity
    vectorizer = TfidfVectorizer(
        tokenizer=lambda s: tokenize(s, args.words_only),
        token_pattern=None,
    )
    X = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(X)

    # Similarity scores
    scores = compute_similarity_scores(sim_matrix)

    print("\nSimilarity scores (off-diagonal cosine):")
    print(f"  Mean:   {scores['mean']:.3f}")
    print(f"  Median: {scores['median']:.3f}")
    print(f"  Core:   {scores['core']:.3f}  (mean of top half of similarities)\n")

    # Pairwise matrix
    print("Pairwise cosine similarity matrix:")
    header = [""] + labels
    print("\t".join(header))
    for i, label in enumerate(labels):
        row = [f"{sim_matrix[i,j]:.3f}" for j in range(len(labels))]
        print("\t".join([label] + row))

    # Per-file similarity to identify outliers
    per_doc = sim_matrix.copy()
    np.fill_diagonal(per_doc, np.nan)
    per_doc_mean = np.nanmean(per_doc, axis=1)

    print("\nPer-file similarity to others:")
    for label, avg in sorted(zip(labels, per_doc_mean), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {avg:.3f}")
    print()

    # Heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(sim_matrix, vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_title(
        "Pairwise Similarity (TF-IDF Cosine)\n"
        f"mean={scores['mean']:.3f}, median={scores['median']:.3f}, core={scores['core']:.3f}"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Similarity")

    plt.tight_layout()
    plt.show(block = False)
    fig.savefig(f"{out_prefix}-heatmap.png", dpi=150)

    # PCA scatter
    X_dense = X.toarray()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_dense)

    fig, ax = plt.subplots()
    ax.scatter(coords[:, 0], coords[:, 1])

    for i, label in enumerate(labels):
        ax.text(coords[i, 0], coords[i, 1], label, ha="center", va="bottom")

    ax.set_title("Outputs in 2D (PCA on TF-IDF)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.show(block = False)
    fig.savefig(f"{out_prefix}-scatter.png", dpi=150)

    plt.show()

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise similarity between LLM outputs stored in text files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Text files containing LLM outputs (wildcards allowed, e.g. 'nse-2025*.txt')."
    )
    parser.add_argument(
        "--words-only",
        action="store_true",
        help="Strip all digits and punctuation before analysis (use words only)."
    )
    args = parser.parse_args()

    main(args)
