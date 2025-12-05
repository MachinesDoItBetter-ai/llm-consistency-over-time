# Similarity Scoring Overview

This project measures how similar a set of LLM outputs are to each other
and provides both visualisations and a single numeric **similarity
score** for easy comparison between datasets or model runs.

## 1. Converting Text to Vectors (TF-IDF)

Each LLM output is treated as a small text document.\
We convert these texts into numeric vectors using **TF-IDF (Term
Frequency--Inverse Document Frequency)**:

-   **Term Frequency (TF):** how often a word appears in a document.\
-   **Inverse Document Frequency (IDF):** how rare that word is across
    all documents.\
-   The result is a weighted vector for each output that emphasizes
    meaningful or distinguishing words.

TF-IDF is: - **lightweight**,\
- **local** (no API calls, no download of large models), and\
- perfectly suited for short structured outputs like "5 words + 5
numbers".

## 2. Computing Pairwise Similarity

Once the outputs are embedded as vectors, we compute **cosine
similarity** between each pair:

\[ `\text{cosine\_similarity}`{=tex}(A, B) =
`\frac{A \cdot B}{\|A\|\|B\|}`{=tex} \]

Meaning: - **1.0** → the outputs are identical in direction (very
similar).\
- **0.0** → they share no detectable similarity.\
- **\< 0** → opposite meaning (rare with TF-IDF for short docs).

This gives us an **N × N similarity matrix**, where `N` is the number of
output files.

## 3. Overall Similarity Score

To make comparisons easy between runs or datasets, we produce a single
scalar value:

> **Similarity Score = mean pairwise cosine similarity between all
> different files (off-diagonal values only).**

We ignore the diagonal (self-similarity = 1.0) so the score reflects
only *real* comparisons.

Formally:

\[ `\text{score}`{=tex} = `\frac{1}{N(N-1)}`{=tex} `\sum`{=tex}\_{i
`\ne `{=tex}j} `\text{sim}`{=tex}(i, j) \]

### Interpretation

-   **High score (e.g., 0.75--0.95):**\
    Outputs are very similar to each other → less diversity.

-   **Medium score (e.g., 0.40--0.70):**\
    Outputs share some patterns but also meaningful variation.

-   **Low score (e.g., 0.00--0.30):**\
    Outputs are very different → high diversity.

This single score allows comparisons between: - different LLM models\
- different temperature settings\
- different prompts\
- different fine-tuning checkpoints\
- different stochastic runs of the same model

## 4. Visualisations

### ● Heatmap

Shows how similar each pair of files is.

### ● PCA 2D Scatter

Reduces TF-IDF vectors to two dimensions to reveal clustering or
outliers.

## 5. Example

If one run prints:

    Overall similarity score: 0.812

and another prints:

    Overall similarity score: 0.544

Then **Run A** is much more internally consistent (or collapsed), while
**Run B** is more diverse.
