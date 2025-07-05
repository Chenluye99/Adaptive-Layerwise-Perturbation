# Local sandbox setup with firejail

Install:

```bash
sudo apt-get update && sudo apt-get install -y firejail
sudo useradd -r -M -s /usr/sbin/nologin sandboxer
pip install "fastapi[all]" uvicorn
```

Run:

```bash
uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 4
```

Test:

```bash
curl -X POST http://127.0.0.1:12345/faas/sandbox/ -H 'Content-Type: application/json' -d '{"code":"print(1+1)","language":"python","compile_timeout":1.0,"run_timeout":3.0}'
# or
SANDBOX_ENDPOINT=http://127.0.0.1:12345/faas/sandbox/ python local_sandbox.py
```
