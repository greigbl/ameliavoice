"""
Word Error Rate (WER) calculation utilities for Japanese ASR evaluation.
"""
import logging
from typing import Dict, List, Tuple, Optional
import jiwer

logger = logging.getLogger(__name__)

# Try to import Japanese tokenizer
_japanese_tokenizer = None
try:
    from janome.tokenizer import Tokenizer
    _japanese_tokenizer = Tokenizer()
    logger.info("Japanese tokenizer (Janome) initialized")
except ImportError:
    logger.warning("Janome not installed. Japanese tokenization will use character-based splitting.")
except Exception as e:
    logger.warning(f"Failed to initialize Japanese tokenizer: {e}")


def _tokenize_japanese(text: str) -> List[str]:
    """
    Tokenize Japanese text into words.
    Falls back to character-based splitting if tokenizer unavailable.
    """
    if not text or not text.strip():
        return []
    
    if _japanese_tokenizer:
        try:
            tokens = _japanese_tokenizer.tokenize(text)
            return [token.surface for token in tokens if token.surface.strip()]
        except Exception as e:
            logger.warning(f"Tokenization error, falling back to character-based: {e}")
    
    # Fallback: split by characters (not ideal but better than space-based)
    # Remove punctuation and split into characters/words
    import re
    # Split on punctuation but keep Japanese characters together
    tokens = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+|[^\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+', text)
    return [t for t in tokens if t.strip()]


def calculate_wer(reference: str, hypothesis: str) -> Dict:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis texts.
    
    WER = (Substitutions + Deletions + Insertions) / Total words in reference
    
    Args:
        reference: Ground truth text (from LLM generation)
        hypothesis: Transcribed text (from ASR model)
    
    Returns:
        Dict with WER metrics including:
        - wer: Word Error Rate (0.0 to 1.0+)
        - wer_percent: WER as percentage
        - substitutions: Number of word substitutions
        - deletions: Number of word deletions
        - insertions: Number of word insertions
        - hits: Number of correct words
        - reference_words: List of reference words
        - hypothesis_words: List of hypothesis words
        - error_words: List of tuples (ref_word, hyp_word, error_type)
    """
    if not reference or not reference.strip():
        return {
            "wer": 1.0,
            "wer_percent": 100.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "hits": 0,
            "reference_words": [],
            "hypothesis_words": [],
            "error_words": [],
            "error": "Reference text is empty"
        }
    
    if not hypothesis or not hypothesis.strip():
        # All words are deletions - tokenize reference properly
        ref_words = _tokenize_japanese(reference)
        return {
            "wer": 1.0,
            "wer_percent": 100.0,
            "substitutions": 0,
            "deletions": len(ref_words),
            "insertions": 0,
            "hits": 0,
            "reference_words": ref_words,
            "hypothesis_words": [],
            "error_words": [(w, None, "deletion") for w in ref_words],
            "error": "Hypothesis text is empty"
        }
    
    # Use jiwer for WER calculation with proper Japanese tokenization
    try:
        # Tokenize Japanese text properly (not space-based)
        ref_words = _tokenize_japanese(reference)
        hyp_words = _tokenize_japanese(hypothesis)
        
        # Join tokens with spaces for jiwer (it expects space-separated words)
        ref_tokenized = " ".join(ref_words)
        hyp_tokenized = " ".join(hyp_words)
        
        # Apply transformations for normalization
        transformation = jiwer.Compose([
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])
        
        # Transform tokenized texts before processing
        ref_transformed = transformation(ref_tokenized)
        hyp_transformed = transformation(hyp_tokenized)
        
        # Use process_words for efficient calculation (newer jiwer API)
        output = jiwer.process_words(ref_transformed, hyp_transformed)
        
        # Extract metrics from output object
        wer = output.wer
        substitutions = output.substitutions
        deletions = output.deletions
        insertions = output.insertions
        hits = output.hits
        
        # Get word-level error mapping for highlighting
        # Map each hypothesis word index to its error type (or None if correct)
        hyp_word_errors = {}
        try:
            alignment = jiwer.align(ref_transformed, hyp_transformed)
            hyp_word_errors = _build_hypothesis_error_map(alignment, hyp_words)
        except:
            # Fallback: use simple alignment
            error_words = _find_error_words(ref_words, hyp_words, substitutions, deletions, insertions)
            hyp_idx = 0
            for ref_word, hyp_word, error_type in error_words:
                if error_type == "substitution" or error_type == "insertion":
                    hyp_word_errors[hyp_idx] = error_type
                    hyp_idx += 1
                elif error_type != "deletion":
                    hyp_idx += 1
        
        return {
            "wer": wer,
            "wer_percent": wer * 100.0,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "hits": hits,
            "reference_words": ref_words,
            "hypothesis_words": hyp_words,
            "hypothesis_word_errors": hyp_word_errors,  # Map of word index -> error type
            "total_reference_words": len(ref_words),
            "total_hypothesis_words": len(hyp_words)
        }
    except Exception as e:
        logger.error(f"WER calculation error: {e}")
        return {
            "wer": 1.0,
            "wer_percent": 100.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "hits": 0,
            "reference_words": [],
            "hypothesis_words": [],
            "error_words": [],
            "error": str(e)
        }


def _build_hypothesis_error_map(alignment, hyp_words: List[str]) -> Dict[int, str]:
    """
    Build a map of hypothesis word indices to error types from jiwer alignment.
    Returns dict mapping word index -> error_type ("substitution" or "insertion")
    """
    hyp_word_errors = {}
    
    try:
        if hasattr(alignment, 'ops'):
            hyp_idx = 0
            for op in alignment.ops:
                if op.type == 'substitute':
                    if hyp_idx < len(hyp_words):
                        hyp_word_errors[hyp_idx] = "substitution"
                    hyp_idx += 1
                elif op.type == 'insert':
                    if hyp_idx < len(hyp_words):
                        hyp_word_errors[hyp_idx] = "insertion"
                    hyp_idx += 1
                elif op.type == 'delete':
                    # Deletions don't map to hypothesis words
                    pass
                elif op.type == 'equal' or op.type == 'match':
                    hyp_idx += 1
    except Exception as e:
        logger.debug(f"Could not extract error map from alignment: {e}")
    
    return hyp_word_errors


def _find_error_words(ref_words: List[str], hyp_words: List[str], 
                      substitutions: int, deletions: int, insertions: int) -> List[Tuple]:
    """
    Find error words by aligning reference and hypothesis using simple word-by-word comparison.
    Returns list of (ref_word, hyp_word, error_type) tuples.
    """
    error_words = []
    
    # Use dynamic programming for better alignment (Levenshtein-like)
    # For now, use simple word-by-word comparison
    ref_idx = 0
    hyp_idx = 0
    
    while ref_idx < len(ref_words) or hyp_idx < len(hyp_words):
        ref_word = ref_words[ref_idx] if ref_idx < len(ref_words) else None
        hyp_word = hyp_words[hyp_idx] if hyp_idx < len(hyp_words) else None
        
        if ref_word is None:
            # Insertion
            error_words.append((None, hyp_word, "insertion"))
            hyp_idx += 1
        elif hyp_word is None:
            # Deletion
            error_words.append((ref_word, None, "deletion"))
            ref_idx += 1
        elif ref_word == hyp_word:
            # Match - skip both
            ref_idx += 1
            hyp_idx += 1
        else:
            # Potential substitution - check if next words match
            if (ref_idx + 1 < len(ref_words) and hyp_idx + 1 < len(hyp_words) and
                ref_words[ref_idx + 1] == hyp_words[hyp_idx + 1]):
                # Next words match, so this is a substitution
                error_words.append((ref_word, hyp_word, "substitution"))
                ref_idx += 1
                hyp_idx += 1
            elif (ref_idx + 1 < len(ref_words) and 
                  ref_words[ref_idx + 1] == hyp_word):
                # Next ref word matches current hyp word - deletion
                error_words.append((ref_word, None, "deletion"))
                ref_idx += 1
            elif (hyp_idx + 1 < len(hyp_words) and
                  ref_word == hyp_words[hyp_idx + 1]):
                # Current ref word matches next hyp word - insertion
                error_words.append((None, hyp_word, "insertion"))
                hyp_idx += 1
            else:
                # Substitution
                error_words.append((ref_word, hyp_word, "substitution"))
                ref_idx += 1
                hyp_idx += 1
    
    return error_words
