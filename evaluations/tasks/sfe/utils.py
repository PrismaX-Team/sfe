import ast
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
import copy
import math

from PIL import Image

import numpy as np
import requests
import yaml
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI

from rouge_score import rouge_scorer
from bert_score import score
import pymeteor.pymeteor as pymeteor

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

import torch

NUM_SECONDS_TO_SLEEP = 5
API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-08-06")
FILE_NAME = os.getenv("FILE_NAME", "sfe_test.json")

JUDGE_RULES = """You are a strict evaluator assessing answer correctness. You must score the model's prediction on a scale from 0 to 10, where 0 represents an entirely incorrect answer and 10 indicates a highly correct answer.
# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{pred}
```


# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Assign a higher score if the predicted answer matches the ground truth, either by option letters or content. Include partial credit for answers that are close in content.
- For exact match and open-ended questions:
  * Assign a high score if the prediction matches the answer semantically, considering variations in format.
  * Deduct points for partially correct answers or those with incorrect additional information.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Scoring Guide
Provide a single integer from 0 to 10 to reflect your judgment of the answer's correctness.

# Strict Output format example
4"""


if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAI(base_url=API_URL, api_key=API_KEY)
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    client = AzureOpenAI(azure_endpoint=API_URL, api_version="2023-07-01-preview", api_key=API_KEY)


scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def get_chat_response(content: str, max_tokens: int, retries: int = 5):
    global MODEL_VERSION
    global client

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and precise assistant for checking the correctness of the answer.",
        },
        {"role": "user", "content": content},
    ]

    payload = {
        "model": MODEL_VERSION,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**payload)
            content = response.choices[0].message.content.strip()
            return content
        except requests.exceptions.RequestException as e:
            eval_logger.warning(f"Request failed on attempt {attempt+1}: {e}")
            time.sleep(NUM_SECONDS_TO_SLEEP)
            if attempt == retries - 1:
                eval_logger.error(f"Failed to get response after {retries} attempts")
                return ""
        except Exception as e:
            eval_logger.error(f"Error on attempt {attempt+1}: {e}")
            return ""


def parse_float_sequence_within(input_str):
    pattern_in_bracket = r"\[(.*)\]"
    match = re.search(pattern_in_bracket, input_str)

    if not match:
        return None

    inside_str = match.group(1)
    groups = inside_str.split(";")

    bboxs = []
    for group in groups:
        floats = group.split(",")
        if len(floats) != 4:
            continue
        try:
            bboxs.append([float(f) for f in floats])
        except Exception as e:
            continue

    if len(bboxs) == 0:
        return None

    return bboxs


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the Intersection over Union
    iou = intersection_area / union_area

    return iou


def greedy_iou(answers, preds):
    score = 0.0
    n_answer, n_pred = len(answers), len(preds)
    selected = []
    for pred in preds:
        if len(selected) == n_answer:
            break
        _scores = [compute_iou(answer, pred) if i not in selected else -1 for i, answer in enumerate(answers)]
        max_index = _scores.index(max(_scores))
        score += max(_scores)
        selected.append(max_index)

    return score / n_answer



def construct_prompt(doc):
    description = f"You are an expert in {doc['field']} and need to solve the following question."
    if doc["question_type"] == "mcq":
        description += "\nThe question is a multiple-choice question. Answer with the option letter from the given choices."
    elif doc["question_type"] == "exact_match":
        description += "\nThe question is an exact match question. Answer the question using a single word or phrase."
    elif doc["question_type"] == "open_ended":
        description += "\nThe question is an open-ended question. Answer the question using a phrase."
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    question = doc["question"]
    question = f"{description}\n\n{question}"
    if doc["question_type"] == "mcq":
        parsed_options = "\n".join(doc["options"])
        question = f"{question}\n{parsed_options}"
    elif doc["question_type"] == "exact_match":
        question = f"{question}"
    elif doc["question_type"] == "open_ended":
        question = f"{question}"
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return question


def sfe_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        question = construct_prompt(doc)
    else:
        question = construct_prompt(doc, lmms_eval_specific_kwargs["multiple_choice_prompt"], lmms_eval_specific_kwargs["open_ended_prompt"], lmms_eval_specific_kwargs["prompt_type"])
    return question


def sfe_doc_to_visual(doc):
    question = construct_prompt(doc)
    images = doc["images"]
    visual = [Image.open(image).convert("RGB") for image in images]
    return visual

def sfe_doc_to_visual_claude(doc):
    images = doc["images"]
    visual = []
    for image in images:
        img = Image.open(image).convert("RGB")
        if max(img.size) > 8000:
            scale = 8000 / max(img.size)
            img = img.resize((min(int(img.size[0] * scale), 8000), min(int(img.size[1] * scale), 8000)), Image.LANCZOS)
        visual.append(img)
    return visual


def sfe_doc_to_visual_doubao(doc):
    images = doc["images"]
    visual = []
    for image in images:
        img = Image.open(image).convert("RGB")
        if img.size[0] * img.size[1] > 36000000:
            scale = 36000000 / (img.size[0] * img.size[1])
            img = img.resize((math.floor(img.size[0] * scale), math.floor(img.size[1] * scale)), Image.LANCZOS)
        visual.append(img)
    return visual


def sfe_process_results(doc, results):
    question_type = doc["question_type"]

    parsed_preds = []

    rough_scores = []
    bertscore_scores = []
    bleu_scores = []
    meteor_scores = []
    llm_scores = []

    execute_success_rate = []
    iou_scores = []

    assert len(results) == 1, f"Expected one result, got {len(results)}"
    for pred in results:
        formatted_question = construct_prompt(doc)
        answer = doc["answer"]

        if doc["id"].split("/")[0].lower() in ["e011", "e012"]:
            answer_bboxs = parse_float_sequence_within(answer)
            pred_bboxs = parse_float_sequence_within(pred)

            if pred_bboxs is not None:
                execute_success_rate.append(1)
                iou_score = greedy_iou(answer_bboxs, pred_bboxs)
                iou_scores.append(iou_score)
            else:
                execute_success_rate.append(0)
                iou_scores.append(-1)
            
            rough_scores.append(-1)
            bertscore_scores.append(-1)
            bleu_scores.append(-1)
            meteor_scores.append(-1)
            llm_scores.append(-1)
        else:
            if question_type == "open_ended":
                try:
                    rouge_score = scorer.score(answer, pred)
                    rough_scores.append(rouge_score["rougeL"].fmeasure)
                except:
                    rough_scores.append(0.)

                try:
                    bertscore = score([answer], [pred], lang="multi", device="cuda" if torch.cuda.is_available() else "cpu")[2].item()
                    bertscore_scores.append(bertscore)
                except:
                    bertscore_scores.append(0.)
                
                try:
                    chencherry = SmoothingFunction()
                    bleu_score = sentence_bleu([answer.strip().split()], pred.strip().split(), smoothing_function=chencherry.method1)
                    bleu_scores.append(bleu_score)
                except:
                    bleu_scores.append(0.)
                
                try:
                    meteor_score = meteor_score([answer.strip().split()], pred.strip().split())
                    meteor_scores.append(meteor_score)
                except:
                    meteor_scores.append(0.)
            else:
                rough_scores.append(-1)
                bertscore_scores.append(-1)
                bleu_scores.append(-1)
                meteor_scores.append(-1)

            # llm_as_a_judge
            llm_judge_prompt = JUDGE_RULES.format(question=formatted_question, answer=answer, pred=pred)
            llm_judge_score = get_chat_response(llm_judge_prompt, max_tokens=20, retries=3)
            llm_scores.append(llm_judge_score)

            execute_success_rate.append(-1)
            iou_scores.append(-1)

        parsed_preds.append(pred)

    all_info = {
        "id": doc["id"], 
        "field": doc["field"], 
        "question_type": doc["question_type"], 
        "answer": doc["answer"], 
        "parsed_pred": parsed_preds, 
        "rouge_score": rough_scores,
        "bertscore": bertscore_scores,
        "bleu_score": bleu_scores,
        "meteor_score": meteor_scores,
        "llm_score": llm_scores,
        "execute_success_rate": execute_success_rate,
        "iou_score": iou_scores,
    }

    rouge_score_info = {
        "id": doc["id"], 
        "field": doc["field"], 
        "question_type": doc["question_type"], 
        "answer": doc["answer"], 
        "parsed_pred": parsed_preds, 
        "rouge_score": rough_scores,
    }

    bert_score_info = {
        "id": doc["id"], 
        "field": doc["field"], 
        "question_type": doc["question_type"], 
        "answer": doc["answer"], 
        "parsed_pred": parsed_preds, 
        "bertscore": bertscore_scores,
    }

    bleu_score_info = {
        "id": doc["id"], 
        "field": doc["field"], 
        "question_type": doc["question_type"], 
        "answer": doc["answer"], 
        "parsed_pred": parsed_preds, 
        "bleu_score": bleu_scores,
    }

    meteor_score_info = {
        "id": doc["id"], 
        "field": doc["field"], 
        "question_type": doc["question_type"], 
        "answer": doc["answer"], 
        "parsed_pred": parsed_preds, 
        "meteor_score": meteor_scores,
    }

    llm_score_info = {
        "id": doc["id"], 
        "field": doc["field"], 
        "question_type": doc["question_type"], 
        "answer": doc["answer"], 
        "parsed_pred": parsed_preds, 
        "llm_score": llm_scores
    }

    execute_succ_rate_info = {
        "id": doc["id"], 
        "field": doc["field"], 
        "question_type": doc["question_type"], 
        "answer": doc["answer"], 
        "parsed_pred": parsed_preds, 
        "execute_success_rate": execute_success_rate,
    }

    iou_score_info = {
        "id": doc["id"], 
        "field": doc["field"], 
        "question_type": doc["question_type"], 
        "answer": doc["answer"], 
        "parsed_pred": parsed_preds, 
        "iou_score": iou_scores,
    }

    return {
        "all_info": all_info, 
        "rouge_score": rouge_score_info, 
        "bert_score": bert_score_info, 
        "bleu_score": bleu_score_info, 
        "meteor_score": meteor_score_info, 
        "llm_score": llm_score_info,
        "execute_succ_rate": execute_succ_rate_info,
        "iou_score": iou_score_info,
        "acc@0.1": iou_score_info,
        "acc@0.3": iou_score_info,
        "acc@0.5": iou_score_info,
        "acc@0.7": iou_score_info,
        "acc@0.9": iou_score_info,
        }


def sfe_save_results(results, args):
    path = os.path.join("/fs-computility/ai4sData/earth-shared/SFE/lmms-eval/examples/sfe/results", FILE_NAME)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {path}.")

    return 0.0


def sfe_aggregate_rouge_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            score = float(result["rouge_score"][0])
            if score < 0:
                continue
            total_score += score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert rouge score to float for {result['id']}: {result['rouge_score'][0]}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1


def sfe_aggregate_bertscore_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            score = float(result["bertscore"][0])
            if score < 0:
                continue
            total_score += score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert bert score to float for {result['id']}: {result['bertscore'][0]}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1


def sfe_aggregate_bleuscore_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            score = float(result["bleu_score"][0])
            if score < 0:
                continue
            total_score += score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert bleu score to float for {result['id']}: {result['bleu_score'][0]}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1


def sfe_aggregate_meteor_score_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            score = float(result["meteor_score"][0])
            if score < 0:
                continue
            total_score += score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert meteor score to float for {result['id']}: {result['meteor_score'][0]}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1
    

def sfe_aggregate_judge_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            item_score = result["llm_score"][0]
            pattern = r"(\d+)"
            match = re.search(pattern, item_score)

            if match:
                item_score = float(match.group(1))
            else:
                item_score = 0

            total_score += item_score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert llm score to int for {result['id']}: {result['llm_score']}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1


def sfe_aggregate_execute_succ_rate_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            score = float(result["execute_success_rate"][0])
            if score < 0:
                continue
            total_score += score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert execute success score to float for {result['id']}: {result['execute_success_rate'][0]}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1


def sfe_aggregate_iou_score_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            score = float(result["iou_score"][0])
            if score < 0:
                continue
            total_score += score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert execute iou score to float for {result['id']}: {result['iou_score'][0]}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1


def sfe_aggregate_acc01_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            score = 1.0 if float(result["iou_score"][0]) > 0.1 else 0.0
            if score < 0:
                continue
            total_score += score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert execute iou score to float for {result['id']}: {result['iou_score'][0]}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1


def sfe_aggregate_acc03_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            score = 1.0 if float(result["iou_score"][0]) > 0.3 else 0.0
            if score < 0:
                continue
            total_score += score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert execute iou score to float for {result['id']}: {result['iou_score'][0]}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1


def sfe_aggregate_acc05_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            score = 1.0 if float(result["iou_score"][0]) > 0.5 else 0.0
            if score < 0:
                continue
            total_score += score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert execute iou score to float for {result['id']}: {result['iou_score'][0]}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1


def sfe_aggregate_acc07_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            score = 1.0 if float(result["iou_score"][0]) > 0.7 else 0.0
            if score < 0:
                continue
            total_score += score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert execute iou score to float for {result['id']}: {result['iou_score'][0]}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1


def sfe_aggregate_acc09_results(results, args):
    total_score = 0
    total_cnt = 0
    for result in results:
        try:
            score = 1.0 if float(result["iou_score"][0]) > 0.9 else 0.0
            if score < 0:
                continue
            total_score += score
            total_cnt += 1
        except:
            eval_logger.warning(f"Failed to convert execute iou score to float for {result['id']}: {result['iou_score'][0]}")
            total_score += 0
    return total_score / total_cnt if total_cnt > 0 else -1
