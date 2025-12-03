# OAS Evaluation Service

This repository provides a small evaluation service that analyzes model responses logged by the orchestrator and the OpenAI agent. The FastAPI endpoint `/eval` runs an evaluation job in the background and sends a webhook callback when the job completes.

Key files and folders
- `app.py` — FastAPI application exposing `/eval`.
- `eval_om.py` — Evaluation logic; loads logs, matches responses to ground truth, optionally uses an LLM-as-judge, and writes `eval_<date>.csv`.
- `logs/` — Stores JSONL logs. Two subfolders: `orchestrator/` and `openai_agent/`.

Required inputs for `/eval`
- `response_id_path`: path to a `.txt` file that contains a list of `response_id`s (one per line). These IDs correspond to entries contained in `logs/openai_agent/responses_<date>.jsonl`.
- `ground_truth_path`: path to a CSV file containing ground truth data (questions, answers, etc.). You can place this file in the same directory that contains `logs/` (the repository root by default). If you have a CSV on Google Drive, download it and set this path to the downloaded file. [Download Ground Truth](https://drive.google.com/file/d/1ckMbY0GGDeK-VQPXHIkS6jjTtLSAAMM2/view?usp=drive_link)

Example `response_ids.txt` (one id per line):
```
resp_12345
resp_23456
resp_34567
```

Expected `ground_truth.csv` columns
- `Question` — the exact question text used during routing
- `Answers` — the reference answer
- `Source_Name` — optional expected source name used for routing checks

Running the evaluation
1. Start the FastAPI app (example):
```pwsh
uvicorn app:app --reload --port 8000
```
2. POST to `/eval` with JSON body matching the `JobRequest` model. Example curl:
```pwsh
curl -X POST "http://localhost:8000/eval" -H "Content-Type: application/json" -d @- <<'JSON'
{
  "response_id_path": "response_ids.txt",
  "ground_truth_path": "single_turn.csv",
  "webhook_url": "https://your-webhook.example/receive"
}
JSON
```

Webhook payload
- On success, the webhook receives JSON like:
```json
{
  "job_id": "...",
  "status": "success",
  "result_file": "eval_YYYY-MM-DD.csv",
  "total_items": 123,
  "duration_sec": 12.34,
  "error": null
}
```
- On failure, `status` will be `failed` and `error` will be populated.

Notes
- Place your `response_ids.txt` and `ground_truth` CSV in the repository root (next to `logs/`) or provide absolute paths.
- If you want the ground truth CSV from Google Drive, download it manually and place it where `ground_truth_path` points.

If you want, I can also add a small helper script to build `response_ids.txt` from existing logs or pull the ground-truth CSV from a Drive link automatically.
