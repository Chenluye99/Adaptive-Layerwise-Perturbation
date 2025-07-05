import asyncio
from sandbox_fusion import (
    RunCodeRequest,
    RunStatus,
    run_code_async,
    set_sandbox_endpoint,
)


async def single_sandbox(
    code, language="python", compile_timeout=1.0, run_timeout=3.0, semaphore=None
):
    request = RunCodeRequest(
        code=code,
        language=language,
        compile_timeout=compile_timeout,
        run_timeout=run_timeout,
    )
    async with semaphore:
        response = await run_code_async(request, client_timeout=30.0, max_attempts=2)
        response = response.dict()
    await asyncio.sleep(2)
    return response


async def parallel_sandbox(sols, num_processes=200):
    semaphore = asyncio.Semaphore(num_processes)
    set_sandbox_endpoint("https://seed-sandbox.byteintl.net/faas/sandbox/")
    tasks_async = [single_sandbox(s, semaphore=semaphore) for s in sols]
    results = await asyncio.gather(*tasks_async, return_exceptions=False)
    return (
        [r["status"] == RunStatus.Success for r in results],
        [r["run_result"]["stdout"] for r in results],
        [r["run_result"]["stderr"] for r in results],
    )
