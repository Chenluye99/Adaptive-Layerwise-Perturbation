"""
Minimal Firejail-based sandbox HTTP API with logging support.
Dependencies:
    pip install "fastapi[all]" uvicorn
System prerequisites:
    * firejail installed (sudo apt install firejail)
    * a non-login user called `sandboxer` (sudo useradd -r -M -s /usr/sbin/nologin sandboxer)
    * a Firejail profile at /etc/firejail/sandbox.profile such as:
        # /etc/firejail/sandbox.profile
        env none                    # start with a clean env
        env keep PATH
        env keep LANG
        env keep PYTHONIOENCODING
        net none                    # disable networking
        private                     # private filesystem rooted at --private dir
        rlimit as 512M              # memory cap
        rlimit cpu 3                # cpu-time cap
        rlimit nproc 50             # process count cap
        caps.drop all               # drop all capabilities
        seccomp
        whitelist /usr/bin/python3
        include /etc/firejail/whitelist-common.inc

Run the service with:
    uvicorn sandbox_api:app --host 0.0.0.0 --port 8000 --workers 4

The API is compatible with the `RunCodeRequest` / `RunResult` schema used by
ByteIntl Seed-Sandbox. A simple curl test:
    curl -X POST http://127.0.0.1:8000/faas/sandbox/ \
         -H 'Content-Type: application/json' \
         -d '{"code":"print(2+2)","language":"python","compile_timeout":1,"run_timeout":3}'

Logging features:
    * All executions are logged to ./sandbox_logs/executions.json
    * Single JSON file containing all execution records
    * View logs: GET /logs/ (list all logs)
    * Get specific log: GET /logs/{execution_id}
"""

import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------- Logging setup ----------------

# 创建日志目录
LOG_DIR = Path("./sandbox_logs")
LOG_DIR.mkdir(exist_ok=True)

# ---------------- Pydantic models ----------------

class RunStatus(str, Enum):
    """Execution outcome."""
    success = "success"
    timeout = "timeout"
    runtime_error = "runtime_error"


class RunCodeRequest(BaseModel):
    """Incoming JSON body from the client."""

    code: str
    stdin: str = ""
    language: str = "python"
    compile_timeout: float = 1.0  # kept for sdk compatibility, unused here
    run_timeout: float = 30.0


class RunResult(BaseModel):
    """JSON response back to the client."""

    status: RunStatus
    run_result: dict
    created_at: datetime


# ---------------- Core runner ----------------

def _save_execution_log(code: str, stdin_data: str, result: dict, execution_id: str):
    """保存执行日志到单个JSON文件"""
    log_entry = {
        "execution_id": execution_id,
        "timestamp": datetime.utcnow().isoformat(),
        "code": code,
        "stdin": stdin_data,
        "result": result
    }
    
    # 保存到单个JSON文件，追加模式
    log_file = LOG_DIR / "executions.json"
    
    # 读取现有数据
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []
    else:
        data = []
    
    # 添加新条目
    data.append(log_entry)
    
    # 写回文件
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def _run_in_firejail(code: str, timeout: float, stdin_data: str = "") -> dict:
    """Execute *code* inside a fresh Firejail sandbox and return stdout/stderr."""

    # 生成执行ID用于日志记录
    execution_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
    
    # 1) Write user code to a tmpfs directory → zero-copy, fast cleanup
    workdir = Path(tempfile.mkdtemp(prefix="fj_", dir="/dev/shm"))
    src = workdir / "main.py"
    src.write_text(code)

    # 2) Build Firejail command line
    cmd = [
        "firejail",
        "--quiet",
        "--profile=/etc/firejail/sandbox.profile",
        #f"--private={workdir}",
        "--net=none",              # disable network
        "--",
        sys.executable,
        src.name,
    ]

    # 3) Strip environment to stay below Firejail's MAX_ENVS=256 limit
    whitelist = ("PATH", "LANG", "LC_ALL", "PYTHONIOENCODING", "TERM")
    clean_env = {k: os.environ[k] for k in whitelist if k in os.environ}

    # 4) Launch subprocess under asyncio, enforce wall-clock timeout
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workdir,
        env=clean_env,
    )

    try:
        input_bytes = (stdin_data + "\n").encode() if len(stdin_data) > 0 else None
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_bytes),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        shutil.rmtree(workdir, ignore_errors=True)
        result = {
            "status": RunStatus.timeout,
            "stdout": "",
            "stderr": "Timeout\n",
        }
        # 保存超时日志
        #_save_execution_log(code, stdin_data, result, execution_id)
        return result

    status = RunStatus.success if proc.returncode == 0 else RunStatus.runtime_error

    # 5) Clean up tmpfs directory
    shutil.rmtree(workdir, ignore_errors=True)

    result = {
        "status": status,
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
    }
    
    # 6) 保存执行日志
    #_save_execution_log(code, stdin_data, result, execution_id)
    
    return result


# ---------------- FastAPI wiring ----------------

app = FastAPI()
POOL = asyncio.Semaphore(200)  # gate per-process concurrency; tune to your CPU


@app.post("/faas/sandbox/", response_model=RunResult)
async def run_code(req: RunCodeRequest):
    """HTTP endpoint: compatible with the Seed-Sandbox client SDK."""

    if req.language != "python":
        raise HTTPException(400, "Only Python is supported in this minimal demo.")

    async with POOL:
        result = await _run_in_firejail(req.code, req.run_timeout, req.stdin)

    return RunResult(status=result["status"], run_result=result, created_at=datetime.utcnow())


@app.get("/logs/")
async def list_logs():
    """列出所有执行日志"""
    log_file = LOG_DIR / "executions.json"
    if not log_file.exists():
        return {"logs": [], "total": 0}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 返回日志摘要信息
        logs_info = []
        for entry in data:
            logs_info.append({
                "execution_id": entry["execution_id"],
                "timestamp": entry["timestamp"],
                "status": entry["result"]["status"],
                "code_preview": entry["code"][:100] + "..." if len(entry["code"]) > 100 else entry["code"]
            })
        
        # 按时间排序，最新的在前
        logs_info.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"logs": logs_info, "total": len(logs_info)}
        
    except (json.JSONDecodeError, FileNotFoundError):
        return {"logs": [], "total": 0}


@app.get("/logs/{execution_id}")
async def get_log(execution_id: str):
    """获取特定执行ID的日志内容"""
    log_file = LOG_DIR / "executions.json"
    if not log_file.exists():
        raise HTTPException(404, "Log file not found")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 查找特定execution_id的条目
        for entry in data:
            if entry["execution_id"] == execution_id:
                return entry
        
        raise HTTPException(404, f"Log entry for execution {execution_id} not found")
        
    except (json.JSONDecodeError, FileNotFoundError):
        raise HTTPException(404, "Log file not found or corrupted")
