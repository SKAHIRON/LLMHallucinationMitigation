# evaluation.py

import os
import torch
import pandas as pd
from tqdm import tqdm

from utils import (
    normalize_answer,
    remove_stop_words,
    compute_f1,
    clean_generated_answer,
    limit_answer_tokens,
)

def evaluate_model(llm, prompts_with_context, prompts_without_context, answers, stop_word_list, args):
    """
    Evaluates the model on given prompts (with and without context),
    then calculates EM (Exact Match), partial match, F1, Precision, and Recall.
    Returns the metrics and details of predictions.
    """
    em_score = 0
    partial_match_score = 0
    f1_score_total = 0
    precision_total = 0
    recall_total = 0
    total_count = len(prompts_with_context)

    incorrect_details = []

    with torch.no_grad():
        for index, (prompt_w_ctx, prompt_no_ctx, true_answer) in enumerate(
            tqdm(zip(prompts_with_context, prompts_without_context, answers), total=total_count)
        ):
            # Generate response from the model
            max_new_tokens = getattr(args, 'max_new_tokens', 25)
            generated_response = llm.generate(
                input_text=prompt_w_ctx,
                input_text2=prompt_no_ctx,
                mode=args.mode,
                alpha=args.alpha,
                layer_alpha=args.layer_alpha,
                start_layer=args.start_layer,
                max_new_tokens=max_new_tokens
            )

            # Post-process generated text to better match gold span answers
            generated_response = clean_generated_answer(generated_response)
            generated_response = limit_answer_tokens(generated_response, max_tokens=8)

            # Normalize gold/predicted answers
            normalized_gen = normalize_answer(generated_response)
            normalized_gold = normalize_answer(true_answer)

            # Remove stopwords
            normalized_gen = remove_stop_words(normalized_gen, stop_word_list)

            # Exact Match
            if normalized_gen == normalized_gold:
                em_score += 1
            elif normalized_gen in normalized_gold:
                partial_match_score += 1

            # Keep record (both correct and incorrect)
            incorrect_details.append({
                "Index": index,
                "True Answer": normalized_gold,
                "Predicted Answer": normalized_gen
            })

            # Compute F1, Precision, Recall
            f1, precision, recall = compute_f1(normalized_gold, normalized_gen)
            f1_score_total += f1
            precision_total += precision
            recall_total += recall

    # Compute average metrics
    em_score /= total_count
    partial_match_score /= total_count
    avg_f1_score = f1_score_total / total_count
    avg_precision = precision_total / total_count
    avg_recall = recall_total / total_count

    return em_score, partial_match_score, avg_f1_score, avg_precision, avg_recall, incorrect_details


def save_incorrect_details(incorrect_details, name):
    """
    Saves the details of predictions to a CSV file (could include correct or incorrect).
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"{name}.csv")
    pd.DataFrame(incorrect_details).to_csv(file_path, index=False)
    print(f"Incorrect predictions saved to {file_path}")
