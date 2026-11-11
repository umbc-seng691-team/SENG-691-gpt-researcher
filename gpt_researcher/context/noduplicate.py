
"""This document was created by the SENG 691 GPT Researchers group as an intensive passage similarity tracker

The noduplicater normalizes text amongst different passages retrieved from GPT Researcher.
Utilizes difflibs imported SequenceMatcher to track similarities between passage sequences and tokens
as well as the Sentence Transformer for semantic similarity. Jaccard similarity to track
similarity and overlap between two or more passages. Containment similarity for overlap coefficient
The remove_duplicate functions works as the main processor of passage similarity and 
deletes passages deemed overly similar based on a specific threshold
Required dependencies that may need to be includes:
    -sentence-transformers==3.4.1
    -scikit-learn
    -torch==2.2.2
    -transformers==4.49.0
    -tokenizers==0.21.1
"""
import re
from difflib import SequenceMatcher

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize_text(text):
    """Normalize text before passages are compared"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_tokens(text):
    """Extract individual words from normalized text."""
    return normalize_text(text).split()


def sequence_similarity(text1, text2):
    """Compare the sequence similarity of two passages."""
    return SequenceMatcher(
        None,
        normalize_text(text1),
        normalize_text(text2)
    ).ratio()


def jaccard_similarity(text1, text2):
    """Calculate similarity and overlap between two passages aka jaccard. |A intersection B| / | A union B |"""
    tokens_1 = set(get_tokens(text1))
    tokens_2 = set(get_tokens(text2))

    if not tokens_1 or not tokens_2:
        return 0.0

    intersection = len(tokens_1.intersection(tokens_2))
    union = len(tokens_1.union(tokens_2))

    return intersection / union


def containment_similarity(text1, text2):
    """Measures for shorter passages overlaps. |A intersection B| / minimum of A & B"""
    tokens_1 = set(get_tokens(text1))
    tokens_2 = set(get_tokens(text2))

    if not tokens_1 or not tokens_2:
        return 0.0

    intersection = len(tokens_1.intersection(tokens_2))

    return intersection / min(len(tokens_1), len(tokens_2))


def semantic_similarity(embedding1, embedding2):
    """Uses semantic search to find differences even instances of similar phrasing from unique text"""
    score = cosine_similarity(
        [embedding1],
        [embedding2]
    )[0][0]

    return max(0.0, float(score))


def calculate_similarity(
    text_1,
    text_2,
    semantic_1=None,
    semantic_2=None
):
    """Calculate sequence, jaccard, containment and semantic similarity."""

    sequence_score = sequence_similarity(text_1, text_2)

    jaccard_score = jaccard_similarity(text_1, text_2)

    containment_score = containment_similarity(text_1, text_2)

    semantic_score = 0.0

    if semantic_1 is not None and semantic_2 is not None:
        semantic_score = semantic_similarity(
            semantic_1,
            semantic_2
        )

    final_score = (
        0.15 * sequence_score +
        0.15 * jaccard_score +
        0.15 * containment_score +
        0.55 * semantic_score
    )

    return final_score


def remove_duplicate_passages(documents, threshold=0.55):
    """Remove similar passages..."""

    if not documents:
        return []

    texts = [doc.page_content for doc in documents]

    # generate all embeddings in one batch.
    embeddings = semantic_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    unique_documents = []
    unique_embeddings = []

    removed_count = 0

    for index, document in enumerate(documents):

        curr_text = document.page_content

        if not curr_text or not curr_text.strip():
            continue

        current_embedding = embeddings[index]

        is_duplicate = False

        for concurrent_index, concurrent_document in enumerate(
            unique_documents
        ):

            existing_embedding = unique_embeddings[concurrent_index]

            similarity = calculate_similarity(
                curr_text,
                concurrent_document.page_content,
                current_embedding,
                existing_embedding
            )

            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_documents.append(document)
            unique_embeddings.append(current_embedding)
        else:
            removed_count += 1


    return unique_documents
