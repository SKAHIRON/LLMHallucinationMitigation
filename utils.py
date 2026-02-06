# utils.py

import random
import numpy as np
import re
import string
import torch
import torch.nn.functional as F
from collections import Counter

# Make wandb optional
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False

##############################################################################
# General Utilities
##############################################################################

def set_seed(seed: int):
    """
    Fixes random seed to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def clear_cuda_cache():
    """
    Clears CUDA cache to free up GPU memory.
    Useful for preventing out-of-memory errors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection
        import gc
        gc.collect()


def get_gpu_memory_info():
    """
    Returns current GPU memory usage information.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, reserved
    return 0, 0


def initialize_wandb(args):
    """
    Initializes a WandB run with the given arguments (no-op if wandb unavailable).
    """
    if not WANDB_AVAILABLE:
        return
    wandb.init(
        project=args.project,
        name=args.name,
        config=vars(args),
        group=args.mode
    )


def log_metrics_to_wandb(em_score, partial_match_score, avg_f1_score, avg_precision, avg_recall):
    """
    Logs performance metrics to WandB (no-op if wandb unavailable).
    """
    if not WANDB_AVAILABLE:
        return
    wandb.log({
        "Exact Match Score": em_score,
        "Partial Match Score": partial_match_score,
        "F1 Score": avg_f1_score,
        "Precision": avg_precision,
        "Recall": avg_recall
    })


def finish_wandb():
    """Finishes a WandB run if available (no-op otherwise)."""
    if WANDB_AVAILABLE:
        try:
            wandb.finish()
        except Exception:
            pass


def log_metrics_to_wandb_summarization(
    rouge1_f, rouge1_p, rouge1_r,
    rouge2_f, rouge2_p, rouge2_r,
    rougel_f, rougel_p, rougel_r
):
    """
    Logs summarization performance metrics to WandB (no-op if wandb unavailable).
    """
    if not WANDB_AVAILABLE:
        return
    wandb.log({
        "ROUGE-1 F1": rouge1_f,
        "ROUGE-1 Precision": rouge1_p,
        "ROUGE-1 Recall": rouge1_r,
        "ROUGE-2 F1": rouge2_f,
        "ROUGE-2 Precision": rouge2_p,
        "ROUGE-2 Recall": rouge2_r,
        "ROUGE-L F1": rougel_f,
        "ROUGE-L Precision": rougel_p,
        "ROUGE-L Recall": rougel_r,
    })


##############################################################################
# Text Normalization & Metric Utilities
##############################################################################

def normalize_answer(s: str) -> str:
    """
    Converts to lowercase, removes punctuation, articles (a/an/the), and extra spaces
    to normalize answers. Improved version for better exact matching.
    """
    if not s:
        return ""
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    
    # Normalize
    normalized = white_space_fix(remove_articles(remove_punc(lower(s))))
    
    # Remove leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized


def compute_f1(gold_answer: str, pred_answer: str):
    """
    Compares gold and predicted answers to compute F1, Precision, and Recall.
    """
    gold_toks = gold_answer.split()
    pred_toks = pred_answer.split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0, 0, 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def remove_stop_words(response: str, stop_word_list: list) -> str:
    """
    Removes specified stopwords from the beginning of the response string.
    """
    for stop_word in stop_word_list:
        if response.startswith(stop_word):
            response = response[len(stop_word):]
    return response


def clean_generated_answer(text: str) -> str:
    """
    Cleans generated answers to improve exact match rate.
    Removes common artifacts, enumerations, and extra whitespace.
    """
    if not text:
        return ""

    original_text = text
    text = text.replace("\r", " ").strip()

    # Remove trailing parts that look like new questions or answers
    for marker in ("\n\nQ:", "\nQ:", "\n\nA:", "\nA:"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()

    # Remove numbering/bullet prefixes such as "1.", "(1)", "1)"
    text = re.sub(r"^\s*[\(\[]?\d+[\)\].:-]?\s*", "", text)

    # Remove common prefixes that might be generated
    prefixes_to_remove = [
        "Answer",
        "A",
        "The answer is",
        "Answer is",
        "It is",
        "It's",
        "That is",
        "That's",
    ]
    lowered = text.lower()
    for prefix in prefixes_to_remove:
        prefix_low = prefix.lower()
        if lowered.startswith(prefix_low):
            cut_len = len(prefix)
            text = text[cut_len:].lstrip(" :.-")
            lowered = text.lower()
            break

    # Collapse multiple whitespaces/newlines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = " ".join(lines)
    else:
        text = ""

    # If cleaning removed everything, fall back to original trimmed text
    if not text:
        text = original_text.strip()

    # Remove duplicated terminal punctuation (e.g., "??", "!!")
    while len(text) > 1 and text[-1] in ".,!?;:" and text[-2] == text[-1]:
        text = text[:-1]

    return text.strip()


def limit_answer_tokens(answer: str, max_tokens: int = 12) -> str:
    """
    Truncates the answer to a reasonable number of tokens to avoid long rambles.
    """
    tokens = answer.split()
    if len(tokens) <= max_tokens:
        return answer

    trimmed = tokens[:max_tokens]
    # Avoid ending on conjunctions/articles
    while trimmed and trimmed[-1].lower() in {"and", "or", "the", "a", "an", "of"}:
        trimmed.pop()
    return " ".join(trimmed).strip()


def extract_question_from_prompt(prompt: str) -> str:
    """
    Extracts the last question from a prompt constructed via create_demo_text.
    """
    marker = "\nQ:"
    idx = prompt.rfind(marker)
    marker_len = len(marker)
    if idx == -1:
        marker = "Q:"
        marker_len = len(marker)
        idx = prompt.rfind(marker)
        if idx == -1:
            return ""
    start = idx + marker_len
    end_marker = "\nA:"
    end_idx = prompt.find(end_marker, start)
    if end_idx == -1:
        end_idx = len(prompt)
    return prompt[start:end_idx].strip()


def extract_supporting_text(prompt: str) -> str:
    """
    Extracts the supporting information block from the prompt, if present.
    """
    marker = "Supporting information:"
    idx = prompt.rfind(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    end_idx = prompt.find("\n\nQ:", start)
    if end_idx == -1:
        end_idx = len(prompt)
    return prompt[start:end_idx].strip()


def split_supporting_sentences(support_text: str) -> list:
    """
    Splits supporting information into coarse sentences for scoring.
    """
    if not support_text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", support_text)
    return [sent.strip() for sent in sentences if sent.strip()]


def is_yes_no_question(question: str) -> bool:
    if not question:
        return False
    starters = {
        "is", "are", "was", "were", "do", "does", "did",
        "has", "have", "had", "can", "could", "should",
        "would", "will", "am", "did", "were", "might",
    }
    first_word = question.strip().split()[0].lower()
    return first_word in starters or question.strip().lower().startswith(("was there", "were there"))


def normalize_yes_no_answer(answer: str) -> str:
    if not answer:
        return answer
    lowered = answer.lower()
    yes_variants = {"yes", "yeah", "yep", "affirmative", "correct"}
    no_variants = {"no", "nope", "nah", "negative"}
    if lowered in yes_variants:
        return "yes"
    if lowered in no_variants:
        return "no"
    if lowered in {"true", "1"}:
        return "yes"
    if lowered in {"false", "0"}:
        return "no"
    return answer


def expand_answer_with_context(answer: str, supporting_text: str, max_extra_tokens: int = 4) -> str:
    """
    Expands short answers with nearby context tokens if they are present in the evidence.
    Improved version for better F1 recall.
    """
    if not answer or not supporting_text:
        return answer
    
    # More lenient - allow expansion for longer answers too
    if len(answer.split()) >= 8:
        return answer

    pattern = re.compile(re.escape(answer), flags=re.IGNORECASE)
    match = pattern.search(supporting_text)
    if not match:
        return answer

    start, end = match.span()
    extended = supporting_text[start:end]

    # Extend forward up to max_extra_tokens, stopping at sentence boundaries.
    tokens_added = 0
    idx = end
    sentence_boundaries = ".!?\n"
    while tokens_added < max_extra_tokens and idx < len(supporting_text):
        if supporting_text[idx] in sentence_boundaries:
            break
        token_match = re.match(r"\s*([^\s,;:!?]+)", supporting_text[idx:])
        if not token_match:
            break
        extension = supporting_text[idx: idx + token_match.end()]
        extended += extension
        idx += token_match.end()
        tokens_added += 1
        # Skip over trailing commas/semicolons without counting as tokens
        while idx < len(supporting_text) and supporting_text[idx] in ",;:":
            extended += supporting_text[idx]
            idx += 1
    
    # Also try extending backward for better context
    if tokens_added < max_extra_tokens:
        idx = start
        backward_tokens = 0
        while backward_tokens < 2 and idx > 0:  # Limit backward expansion
            idx -= 1
            if supporting_text[idx] in sentence_boundaries:
                break
            # Look for word boundary going backward
            if idx > 0 and supporting_text[idx-1] in " \t\n":
                word_start = idx
                while word_start > 0 and supporting_text[word_start-1] not in " \t\n":
                    word_start -= 1
                if word_start < idx:
                    extension = supporting_text[word_start:start]
                    extended = extension + extended
                    backward_tokens += 1
                    start = word_start

    return extended.strip()


def extract_answer_span_from_context(generated_answer: str, supporting_text: str, question: str = "") -> str:
    """
    Tries to find the best matching span in the context that matches the generated answer.
    This helps when the model generates a paraphrase or partial answer.
    Improved version with better span extraction.
    """
    if not generated_answer or not supporting_text:
        return generated_answer
    
    # Normalize for matching
    gen_normalized = normalize_answer(generated_answer)
    if not gen_normalized:
        return generated_answer
    
    gen_words = gen_normalized.split()
    if not gen_words:
        return generated_answer
    
    # Try direct substring matching first (case-insensitive)
    gen_lower = generated_answer.lower().strip()
    supporting_lower = supporting_text.lower()
    
    # Look for exact substring match
    if gen_lower in supporting_lower:
        idx = supporting_lower.find(gen_lower)
        if idx != -1:
            # Extract from original (preserving case)
            extracted = supporting_text[idx:idx+len(generated_answer)].strip()
            if extracted:
                return extracted
    
    # Split context into sentences and phrases
    sentences = split_supporting_sentences(supporting_text)
    if not sentences:
        sentences = [supporting_text]
    
    best_match = generated_answer
    best_score = 0.0
    
    # Check each sentence for matches
    for sent in sentences:
        sent_normalized = normalize_answer(sent)
        if not sent_normalized:
            continue
        
        sent_words = sent_normalized.split()
        
        # Exact normalized match
        if gen_normalized == sent_normalized:
            return sent  # Perfect match, return the sentence
        elif gen_normalized in sent_normalized:
            # Generated answer is substring of sentence
            if len(sent) < len(best_match) or best_score < 0.9:
                best_match = sent
                best_score = 0.95
                continue
        elif sent_normalized in gen_normalized:
            # Sentence is substring of generated answer
            if len(sent) < len(best_match):
                best_match = sent
                best_score = 0.9
                continue
        
        # Token overlap score
        gen_tokens = set(gen_words)
        sent_tokens = set(sent_words)
        
        if not gen_tokens:
            continue
        
        overlap = len(gen_tokens & sent_tokens)
        if overlap > 0:
            # Use F1-like score: harmonic mean of precision and recall
            precision = overlap / len(gen_tokens)
            recall = overlap / len(sent_tokens) if sent_tokens else 0
            if precision > 0 and recall > 0:
                score = 2 * precision * recall / (precision + recall)
            else:
                score = overlap / max(len(gen_tokens), 1)
            
            if score > best_score and score > 0.55:  # Lower threshold for better matching
                best_match = sent
                best_score = score
    
    # If we found a good match, try to extract just the answer part
    if best_score > 0.65 and len(gen_words) <= 12:  # More lenient threshold
        # Try to find a shorter span within the best match
        best_normalized = normalize_answer(best_match)
        best_words = best_normalized.split()
        
        # Find the best contiguous span that contains most of the answer words
        if len(gen_words) <= len(best_words):
            best_span = None
            best_span_score = 0
            
            # Try different window sizes
            for window_size in range(len(gen_words), min(len(gen_words) + 3, len(best_words) + 1)):
                for i in range(len(best_words) - window_size + 1):
                    span_words = best_words[i:i+window_size]
                    span_set = set(span_words)
                    gen_set = set(gen_words)
                    overlap = len(span_set & gen_set)
                    if overlap > 0:
                        precision = overlap / len(gen_words)
                        recall = overlap / len(span_words)
                        if precision > 0 and recall > 0:
                            span_score = 2 * precision * recall / (precision + recall)
                        else:
                            span_score = overlap / max(len(gen_words), 1)
                        
                        if span_score > best_span_score and span_score > 0.7:
                            best_span = " ".join(span_words)
                            best_span_score = span_score
            
            if best_span and best_span_score > 0.7:
                # Try to find this span in the original sentence
                best_span_lower = best_span.lower()
                best_match_lower = best_match.lower()
                idx = best_match_lower.find(best_span_lower)
                if idx != -1:
                    # Extract preserving original case
                    extracted = best_match[idx:idx+len(best_span)].strip()
                    if extracted:
                        return extracted
        
        return best_match
    
    return generated_answer


def find_best_answer_candidate(candidates: list, supporting_text: str, question: str = "") -> str:
    """
    Selects the best answer from multiple candidates using various heuristics.
    """
    if not candidates:
        return ""
    
    # Remove empty candidates
    candidates = [c for c in candidates if c and c.strip()]
    if not candidates:
        return ""
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Score each candidate
    scored = []
    for cand in candidates:
        score = 0.0
        cand_norm = normalize_answer(cand)
        
        # Prefer shorter answers (more precise)
        if len(cand_norm.split()) <= 10:
            score += 0.3
        
        # Prefer answers that appear in context
        if supporting_text:
            if cand_norm.lower() in supporting_text.lower():
                score += 0.4
            elif any(word in supporting_text.lower() for word in cand_norm.split() if len(word) > 3):
                score += 0.2
        
        # Penalize very long answers
        if len(cand_norm.split()) > 20:
            score -= 0.2
        
        # Prefer answers that don't look like questions
        if not cand.strip().endswith("?"):
            score += 0.1
        
        scored.append((score, cand))
    
    # Sort by score and return best
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def post_process_answer(raw_answer: str, question: str = "", supporting_text: str = "") -> str:
    """
    Minimal post-processing - just basic cleaning to avoid breaking correct answers.
    """
    if not raw_answer:
        return ""
    
    # Very minimal cleaning - just remove leading/trailing whitespace and newlines
    cleaned = raw_answer.strip().replace('\n', ' ').replace('\r', ' ')
    # Collapse multiple spaces
    cleaned = ' '.join(cleaned.split())
    
    return cleaned


##############################################################################
# Prompt Creation Helpers
##############################################################################

def create_demo_text() -> str:
    """
    Creates a demonstration Q&A text that will be used by the model as examples.
    """
    questions = [
        "Is Mars called the Red Planet?",
        "What is the tallest mountain in the world?",
        "Who wrote the play 'Romeo and Juliet'?",
        "What is the capital city of Australia?",
        "Which element has the chemical symbol 'O'?",
        "Was the Mona Lisa painted by Leonardo da Vinci?"
    ]
    answers = [
        "yes.",
        "Mount Everest.",
        "William Shakespeare.",
        "Canberra.",
        "Oxygen.",
        "yes"
    ]
    demo_text = (
        "Interpret each question literally, and as a question about the real world; "
        "carefully research each answer, without falling prey to any common myths; "
        "and reply \"I have no comment\" unless you are completely certain of the answer. "
        "And you can get information from Supporting information.\n\n"
    )
    for i in range(len(questions)):
        demo_text += f"Q: {questions[i]}\nA: {answers[i]}\n"
    return demo_text


def create_summarization_demo_text() -> str:
    """
    Creates a demonstration summarization text that will be used by the model as examples.
    """
    examples = [
        {
            "text": "The weather today is sunny with temperatures reaching 25 degrees Celsius. Many people are enjoying outdoor activities in the park.",
            "summary": "Sunny weather with 25°C temperatures encourages outdoor activities."
        },
        {
            "text": "Scientists have discovered a new species of butterfly in the Amazon rainforest. The butterfly has unique blue and green patterns on its wings.",
            "summary": "New butterfly species with blue-green wing patterns found in Amazon."
        },
        {
            "text": "The local library will be closed next Monday for renovations. It will reopen on Tuesday with extended hours.",
            "summary": "Library closed Monday for renovations, reopens Tuesday with extended hours."
        }
    ]
    
    demo_text = (
        "You are an expert at creating concise, informative summaries. "
        "Focus on the main points and key information. "
        "Keep summaries clear and factual.\n\n"
    )
    
    for example in examples:
        demo_text += f"Text: {example['text']}\nSummary: {example['summary']}\n\n"
    
    return demo_text


def extract_supporting_facts(context: dict, supporting_facts: dict):
    """
    Used for HotpotQA. For each title and sent_id in supporting_facts,
    extracts the corresponding sentences from the provided context.
    """
    extracted_facts = []
    if context and supporting_facts:
        titles = context.get('title', [])
        sentences = context.get('sentences', [])

        for title, sent_ids in zip(supporting_facts.get('title', []), supporting_facts.get('sent_id', [])):
            if title in titles:
                title_index = titles.index(title)
                if isinstance(sent_ids, list):
                    # If sent_ids is a list, extract multiple sentences
                    combined_sentences = []
                    for sid in sent_ids:
                        if sid < len(sentences[title_index]):
                            combined_sentences.append(sentences[title_index][sid])
                    extracted_facts.append(' '.join(combined_sentences))
                elif isinstance(sent_ids, int):
                    # If sent_ids is a single integer
                    if sent_ids < len(sentences[title_index]):
                        extracted_facts.append(sentences[title_index][sent_ids])
    return extracted_facts


##############################################################################
# Jensen-Shannon Divergence
##############################################################################

def compute_rouge_scores(reference: str, candidate: str) -> dict:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores between reference and candidate summaries.
    
    Args:
        reference: Reference summary text
        candidate: Generated summary text
        
    Returns:
        Dictionary with 'rouge1', 'rouge2', 'rougel' scores (each with 'f', 'p', 'r')
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        return {
            'rouge1': {
                'f': scores['rouge1'].fmeasure,
                'p': scores['rouge1'].precision,
                'r': scores['rouge1'].recall
            },
            'rouge2': {
                'f': scores['rouge2'].fmeasure,
                'p': scores['rouge2'].precision,
                'r': scores['rouge2'].recall
            },
            'rougel': {
                'f': scores['rougeL'].fmeasure,
                'p': scores['rougeL'].precision,
                'r': scores['rougeL'].recall
            }
        }
    except ImportError:
        print("Warning: rouge_score not available. Install with: pip install rouge-score", flush=True)
        return {
            'rouge1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rougel': {'f': 0.0, 'p': 0.0, 'r': 0.0}
        }


def jensen_shannon_divergence(probs_p: torch.Tensor, probs_q: torch.Tensor) -> torch.Tensor:
    """
    Computes Jensen-Shannon Divergence between two probability distributions
    represented as PyTorch tensors. (Must be of same shape.)
    """
    # Convert to double for numerical stability
    p = probs_p.double()
    q = probs_q.double()

    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log(), p, reduction='batchmean')
    kl_qm = F.kl_div(m.log(), q, reduction='batchmean')
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd.sqrt()  # Using sqrt of JS divergence for a distance-like measure
