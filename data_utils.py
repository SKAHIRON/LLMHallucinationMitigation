# data_utils.py

import os
import json
from datasets import load_dataset
from utils import create_demo_text, extract_supporting_facts, create_summarization_demo_text

def load_data_and_create_prompts(args):
    """
    Loads the specified dataset and creates prompts (with and without context)
    along with the answers/summaries.
    Returns: (prompts_with_context, prompts_without_context, answers/summaries)
    """
    task_type = getattr(args, 'task_type', 'qa')  # Default to 'qa' for backward compatibility
    
    if task_type == 'summarization':
        # Summarization datasets
        if args.dataset == 'cnn_dailymail':
            dataset_dict = load_dataset("cnn_dailymail", "3.0.0")
            validation_dataset = dataset_dict["validation"].select(range(50))
            prompts_with_context, summaries = create_prompts_from_cnn_dailymail(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_cnn_dailymail(validation_dataset, include_context=False)
        
        elif args.dataset == 'xsum':
            dataset_dict = load_dataset("xsum")
            validation_dataset = dataset_dict["validation"].select(range(500))
            prompts_with_context, summaries = create_prompts_from_xsum(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_xsum(validation_dataset, include_context=False)
        
        elif args.dataset == 'samsum':
            dataset_dict = load_dataset("samsum")
            validation_dataset = dataset_dict["validation"].select(range(500))
            prompts_with_context, summaries = create_prompts_from_samsum(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_samsum(validation_dataset, include_context=False)
        
        else:
            raise ValueError(f"Invalid summarization dataset name. Choose from 'cnn_dailymail', 'xsum', or 'samsum'.")
        
        return prompts_with_context, prompts_without_context, summaries
    
    else:
        # QA datasets (original functionality)
        if args.dataset == 'hotpot_qa':
            dataset_dict = load_dataset("hotpot_qa", "distractor")
            validation_dataset = dataset_dict["validation"].select(range(1000))
            prompts_with_context, answers = create_prompts_from_hotpot(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_hotpot(validation_dataset, include_context=False)

        elif args.dataset == 'squad':
            dataset_dict = load_dataset("rajpurkar/squad", "plain_text")
            validation_dataset = dataset_dict["validation"].select(range(500))
            prompts_with_context, answers = create_prompts_from_squad(validation_dataset, include_context=True)
            prompts_without_context, _ = create_prompts_from_squad(validation_dataset, include_context=False)

        elif args.dataset == 'strategyqa':
            # StrategyQA: load from a local JSON file only (no HTTP calls inside the code).
            # Preferred path order: explicit --strategyqa-path, then default 'data/strategyqa/dev.json'
            # from the official repo: https://github.com/eladsegal/strategyqa/tree/main/data/strategyqa
            data_list = None
            last_err = None

            # 1) Explicit path from arguments
            local_path = getattr(args, "strategyqa_path", None)
            if not local_path:
                # 2) Default conventional location
                local_path = os.path.join("data", "strategyqa", "dev.json")

            try:
                if not os.path.exists(local_path):
                    raise FileNotFoundError(
                        f"StrategyQA file not found at '{local_path}'. "
                        "Download 'dev.json' from the official repo "
                        "https://github.com/eladsegal/strategyqa (data/strategyqa/dev.json) "
                        "and place it there, or pass --strategyqa-path to its location."
                    )
                with open(local_path, "r", encoding="utf-8") as f:
                    data_list = json.load(f)
            except Exception as e:
                last_err = e
                data_list = None

            if data_list is None:
                raise ValueError(
                    f"StrategyQA failed to load from local file. Last error: {last_err}"
                )

            # Normalize possible formats: expect a list of QA dicts
            if isinstance(data_list, dict):
                # Common keys in StrategyQA repo
                for key in ["data", "examples", "questions"]:
                    if key in data_list:
                        data_list = data_list[key]
                        break

            if not isinstance(data_list, list):
                raise ValueError("StrategyQA JSON format not understood: expected a list of examples.")

            # Use up to 400 examples for evaluation
            eval_data = data_list[:400]
            prompts_with_context, answers = create_prompts_from_strategyqa(eval_data, include_context=True)
            prompts_without_context, _ = create_prompts_from_strategyqa(eval_data, include_context=False)

        else:
            raise ValueError("Invalid dataset name. Choose from 'hotpot_qa', 'squad', or 'strategyqa'.")

        return prompts_with_context, prompts_without_context, answers


def create_prompts_from_hotpot(dataset, include_context: bool):
    """
    Creates prompts and answers for HotpotQA.
    If include_context=True, attaches supporting facts to the prompts.
    """
    prompts = []
    answers = []

    for item in dataset:
        question = item['question']
        answer = item['answer']
        context = item.get('context', {})
        supporting_facts = item.get('supporting_facts', {})

        # Build prompt with an explicit instruction to return only the short answer span.
        instruction = create_demo_text() + " Answer with only the short answer (a single word or short phrase), no explanation.\n"
        if include_context:
            facts = extract_supporting_facts(context, supporting_facts)
            if facts:
                facts_str = ' '.join(facts)
                prompt = f"{instruction}Supporting information: {facts_str}\n\nQ: {question}\nA: "
            else:
                prompt = f"{instruction}\n\nQ: {question}\nA: "
        else:
            prompt = f"{instruction}\n\nQ: {question}\nA: "

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers


def create_prompts_from_squad(dataset, include_context: bool):
    """
    Creates prompts and answers for SQuAD.
    If include_context=True, attaches the paragraph context to the prompts.
    """
    prompts = []
    answers = []

    for item in dataset:
        question = item['question']
        # 'answers' is a list in SQuAD; take the first one
        answer = item['answers']['text'][0]
        context = item['context']

        instruction = create_demo_text()
        if include_context:
            prompt = f"{instruction}Supporting information: {context}\n\nQ: {question}\nA: "
        else:
            prompt = f"{instruction}\n\nQ: {question}\nA: "

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers


def create_prompts_from_strategyqa(dataset, include_context: bool):
    """
    Creates prompts and answers for StrategyQA (yes/no).
    If include_context=True, attaches provided facts as supporting information.
    """
    prompts = []
    answers = []

    for item in dataset:
        question = item.get("question", "")
        facts = item.get("facts", [])
        # StrategyQA answers are booleans; normalize to yes/no strings
        raw_answer = item.get("answer", False)
        if isinstance(raw_answer, bool):
            answer = "yes" if raw_answer else "no"
        elif isinstance(raw_answer, str):
            answer = "yes" if raw_answer.strip().lower() in {"yes", "true", "1"} else "no"
        else:
            answer = "no"

        instruction = create_demo_text()
        if include_context and facts:
            facts_str = " ".join(facts) if isinstance(facts, list) else str(facts)
            prompt = f"{instruction}Supporting information: {facts_str}\n\nQ: {question}\nA: "
        else:
            prompt = f"{instruction}\n\nQ: {question}\nA: "

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers


# Summarization dataset loaders

def create_prompts_from_cnn_dailymail(dataset, include_context: bool):
    """
    Creates prompts and summaries for CNN/DailyMail.
    If include_context=True, attaches the article to the prompts.
    """
    prompts = []
    summaries = []

    for item in dataset:
        article = item['article']
        highlights = item['highlights']  # This is the reference summary
        
        instruction = create_summarization_demo_text()
        if include_context:
            prompt = f"{instruction}Article: {article}\n\nSummary: "
        else:
            prompt = f"{instruction}Summary: "

        prompts.append(prompt)
        summaries.append(highlights)

    return prompts, summaries


def create_prompts_from_xsum(dataset, include_context: bool):
    """
    Creates prompts and summaries for XSum.
    If include_context=True, attaches the document to the prompts.
    """
    prompts = []
    summaries = []

    for item in dataset:
        document = item['document']
        summary = item['summary']
        
        instruction = create_summarization_demo_text()
        if include_context:
            prompt = f"{instruction}Document: {document}\n\nSummary: "
        else:
            prompt = f"{instruction}Summary: "

        prompts.append(prompt)
        summaries.append(summary)

    return prompts, summaries


def create_prompts_from_samsum(dataset, include_context: bool):
    """
    Creates prompts and summaries for SamSum (dialogue summarization).
    If include_context=True, attaches the dialogue to the prompts.
    """
    prompts = []
    summaries = []

    for item in dataset:
        dialogue = item['dialogue']
        summary = item['summary']
        
        instruction = create_summarization_demo_text()
        if include_context:
            prompt = f"{instruction}Dialogue: {dialogue}\n\nSummary: "
        else:
            prompt = f"{instruction}Summary: "

        prompts.append(prompt)
        summaries.append(summary)

    return prompts, summaries
