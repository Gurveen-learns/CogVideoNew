import time
import runpod

print("BOOT: serverless worker started", flush=True)

def handler(job):
    # echo back input so you can confirm requests work
    return {"ok": True, "input": job.get("input", {})}

runpod.serverless.start({"handler": handler})

# keep process alive so logs always exist
while True:
    time.sleep(60)
