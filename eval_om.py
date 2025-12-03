import json
import csv
import pandas as pd
from pathlib import Path
from datetime import date
from typing import Dict, List, Optional
import os
from openai import OpenAI
import logging
from dotenv import load_dotenv
load_dotenv()

LOGS_DIR = Path("logs")
ROUTING_LOG_DIR = LOGS_DIR / "orchestrator"
RESPONSE_LOG_DIR = LOGS_DIR / "openai_agent"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL","gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

LLM_JUDGE_ENABLED = True

LLM_JUDGE_CRITERIA = [
    "correctness",
    "relevance",
]

LLM_JUDGE_INSTRUCTION = """
You are an expert evaluator. Score the AI-generated answer compared to the reference answer.

For each criterion, give a score from 0 to 5:
- correctness: factual correctness (0=completely wrong, 5=perfectly correct)
- relevance: how relevant the answer is to the question (0=irrelevant, 5=highly relevant)

Return ONLY valid JSON in the following format:
{
  "correctness": <0-5>,
  "relevance": <0-5>
}
"""

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)
    

def get_today_log_files():
    """Get today's log file paths"""
    today = date.today()
    routing_file = ROUTING_LOG_DIR / f"routing_{today}.jsonl"
    response_file = RESPONSE_LOG_DIR / f"responses_{today}.jsonl"
    return routing_file, response_file


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file into list of dictionaries"""
    data = []
    if not file_path.exists():
        logger.info(f"File not found: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_ground_truth(csv_file: Path = Path("single_turn.csv")) -> Dict:
    """Load ground truth from single_turn.csv"""
    if not csv_file.exists():
        logger.info(f"Ground truth file not found: {csv_file}")
        return {}
    
    df = pd.read_csv(csv_file)

    ground_truth = {}
    
    for _, row in df.iterrows():
        question = str(row.get('Question', '')).strip()
        answer = str(row.get('Answers', '')).strip()
        source_name = str(row.get('Source_Name', '')).strip()
        
        if question:
            ground_truth[question] = {
                "answer": answer,
                "source_name": source_name
            }
    
    return ground_truth


def empty_judge_scores() -> dict:
    """Create empty dict for all LLM-as-Judge criteria"""
    scores = {}
    for c in LLM_JUDGE_CRITERIA:
        scores[f"judge_{c}"] = None
    scores['judge_usage_input'] = None
    scores['judge_usage_output'] = None
    return scores


def judge_with_llm(question: str, reference: str, hypothesis: str) -> dict:
    """Evaluate AI-generated answer using OpenAI official SDK"""
    
    if not LLM_JUDGE_ENABLED:
        return empty_judge_scores()
    
    try:
        user_payload = {
            "question": question,
            "reference_answer": reference,
            "generated_answer": hypothesis
        }
        
        completion = client.chat.completions.create(
            model=LLM_JUDGE_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": LLM_JUDGE_INSTRUCTION},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ]
        )
        
        content = completion.choices[0].message.content
        raw_scores = json.loads(content)
        
        judge_result = {}
        for c in LLM_JUDGE_CRITERIA:
            judge_result[f"judge_{c}"] = raw_scores.get(c)
        
        judge_result['judge_usage_input'] = completion.usage.prompt_tokens
        judge_result['judge_usage_output'] = completion.usage.completion_tokens
        
        return judge_result
    
    except Exception as e:
        logger.info(f"LLM-as-Judge error: {e}")
        return empty_judge_scores()


def evaluate_logs(routing_file: Path, response_file: Path, ground_truth: Dict,response_ids: Optional[List]):
    """Main evaluation function"""
    if len(response_ids) == 0:
        logger.info('There is no response id')
        return
    routing_logs = load_jsonl(routing_file)
    response_logs = load_jsonl(response_file)

    routing_lookup = {r['orchestrator_request_id']: r for r in routing_logs}
    
    conversations = {}
    for resp in response_logs:
        orch_id = resp.get('orchestrator_request_id')
        response_id = resp.get('response_id')
        
        if response_id not in conversations:
            conversations[response_id] = []
        
        conversations[response_id].append({
            'response': resp,
            'routing': routing_lookup.get(orch_id)
        })
    
    results = []
    conv_idx = -1
    for _, turns in conversations.items():
        for _, turn_data in enumerate(turns, start=1):
            resp = turn_data['response']
            routing = turn_data['routing']
            if not resp['previous_response_id']:
                conv_idx += 1
                turn = 1
            else:
                turn += 1
            response_id = resp.get('response_id', '')
            user_query = routing.get('question', '')
            generated_answer = resp.get('assistant_response', '')
            
            gt = ground_truth.get(user_query, {})
            reference_answer = gt.get('answer', '')
            expected_source = gt.get('source_name', '')
            
            selected_sources = routing.get('selected_sources', []) if routing else []
            routing_correct = expected_source in selected_sources if expected_source else None
            
            judge_scores = {}
            error = None
            
            if reference_answer and generated_answer:
                try:
                    judge_scores = judge_with_llm(user_query, reference_answer, generated_answer)
                except Exception as e:
                    error = str(e)
                    judge_scores = empty_judge_scores()
            else:
                judge_scores = empty_judge_scores()
                if not reference_answer:
                    error = "No ground truth available"
            
            result = {
                'response_id': response_id,
                'conversation_id': conv_idx,
                'turn': turn,
                'question': user_query,
                'reference_answer': reference_answer,
                'generated_answer': generated_answer,
                'error': error or '',
                'routing_correct': routing_correct,
                'expected_sources': expected_source,
                'selected_sources': ','.join(selected_sources) if selected_sources else '',
                'routing_decision': routing.get('decision', '') if routing else '',
                'routing_reasoning': routing.get('reasoning', '') if routing else '',
                'routing_model': routing.get('model', '') if routing else '',
                'judge_correctness': judge_scores.get('judge_correctness'),
                'judge_relevance': judge_scores.get('judge_relevance'),
                'judge_usage_input': judge_scores.get('judge_usage_input'),
                'judge_usage_output': judge_scores.get('judge_usage_output'),
            }
            
            results.append(result)
    
    if results:
        fieldnames = [
            'response_id', 'conversation_id', 'turn', 'question', 
            'reference_answer', 'generated_answer', 'error',
            'routing_correct', 'expected_sources', 'selected_sources',
            'routing_decision', 'routing_reasoning', 'routing_model',
            'judge_correctness', 'judge_relevance',
            'judge_usage_input', 'judge_usage_output'
        ]
        today = date.today()
        output_csv = f"eval_{today}.csv"
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Evaluation complete! Results saved to: {output_csv}")
        logger.info(f"Total evaluations: {len(results)}")
    else:
        logger.info("No results to write")
    
    return results
def load_response_id(response_id_path: Path):
    with open(response_id_path,'r') as file:
        response_ids = [line.strip() for line in file.readlines()]
    return response_ids
def eval_om(response_id_path: Path,ground_truth_path: Path = Path('single_turn.csv')):
    response_ids = load_response_id(response_id_path)
    if len(response_ids) == 0:
        logger.info('There is no response id')
        return
    ground_truth = load_ground_truth(ground_truth_path)
    routing_file, response_file = get_today_log_files()
    evaluate_logs(routing_file, response_file, ground_truth,response_ids)