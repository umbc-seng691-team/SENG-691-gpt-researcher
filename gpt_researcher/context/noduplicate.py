"""This document was created by the SENG 691 GPT Researchers group as an intensive passage similarity tracker

The noduplicater normalizes text amongst different passages retrieved from GPT Researcher.
Utilizes difflibs imported SequenceMatcher to track similarities between passage sequences and tokens
The remove_duplicate functions works as the main processor of passage similarity and 
deletes passages deemed overly similar based on a specific threshold.

"""

from difflib import SequenceMatcher
import re

# simplify matching process
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_similarity(text1, text2):
    """
    Compares passages using sequence and token similarity.
    """
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0.0

    # char-level similarity
    sequence_similarity = SequenceMatcher(
        None, text1, text2
    ).ratio()

    # word-level similarity
    words1 = set(text1.split())
    words2 = set(text2.split())

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    word_similarity = len(intersection) / len(union) if union else 0.0

    return max(sequence_similarity, word_similarity)


def remove_duplicate_passages(documents, threshold=0.65):
    """
    Remove highly similar passages while preserving
    original Document objects and metadata.
    """
    unique_docs = []

    for document in documents:
        context = normalize_text(document.page_content)

        if not context:
            continue

        is_duplicate = False

        # loops through docs until a similar passage pass recommended threshold is found
        for existing in unique_docs:
            similarity = get_similarity(
                context,
                existing.page_content
                
            )

            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_docs.append(document)


    return unique_docs
