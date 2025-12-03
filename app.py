import asyncio
import logging
import os
import time
from datetime import date
from typing import Any, Literal, Optional
from typing import Dict
from typing import List
from typing import Union
from typing import Tuple
from uuid import uuid4
import httpx
from fastapi import FastAPI, Request, File, Form, HTTPException, Query, BackgroundTasks
from eval_om import eval_om
from pydantic import BaseModel
from pathlib import Path


logging.basicConfig(level=os.environ.get('LOGLEVEL','INFO').upper())
logger = logging.getLogger(__name__)

app = FastAPI()

class JobRequest(BaseModel):
    response_id_path: Path
    ground_truth_path: Path
    webhook_url: str

async def send_webhook_callback(webhook_url: str, payload: Dict):
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(webhook_url,json=payload)
            logger.info(f"Webhook delivered: {resp.status_code}")
        except Exception as e:
            print("Webhook failed:", e)
@app.post('/eval')
async def eval(request: JobRequest, background_tasks: BackgroundTasks):
    """
    Run evaluation asynchronously.
    After finish, trigger webhook callback with result summary.
    """
    job_id = str(uuid4())

    # Convert Path input (ground_truth_path) → real Path object
    gt_path = Path(request.ground_truth_path)

    logger.info(f"[EVAL] Received job {job_id}")
    logger.info(f"[EVAL] Response IDs Path: {request.response_id_path}")
    logger.info(f"[EVAL] Ground truth file: {gt_path}")
    logger.info(f"[EVAL] Webhook: {request.webhook_url}")

    async def run_eval_job():
        start = time.time()
        status = "success"
        result_csv_path = None
        total_items = 0
        error_msg = None

        try:
            results = eval_om(
                response_id_path=request.response_id_path,
                ground_truth_path=gt_path
            )

            today = date.today()
            result_csv_path = f"eval_{today}.csv"

            if results:
                total_items = len(results)
            else:
                total_items = 0

        except Exception as e:
            status = "failed"
            error_msg = str(e)
            logger.error(f"[EVAL] Job {job_id} failed: {e}")

        end = time.time()

        # ---- Build webhook payload ----
        payload = {
            "job_id": job_id,
            "status": status,
            "result_file": result_csv_path,
            "total_items": total_items,
            "duration_sec": round(end - start, 2),
            "error": error_msg,
        }

        # ---- Send webhook ----
        await send_webhook_callback(request.webhook_url, payload)

    # Run in background
    background_tasks.add_task(run_eval_job)

    return {
        "job_id": job_id,
        "status": "started",
        "message": "Evaluation job is running in background. Webhook will notify when finished."
    }