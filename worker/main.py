import logging,time
from app.services.queue import Queue
from worker.processor import process

logging.basicConfig(level=logging.INFO)
log=logging.getLogger("signature-worker")

def main():
    q=Queue(); log.info("Worker started")
    while True:
        message,token=q.receive()
        if message is None:
            time.sleep(2); continue
        try:
            process(message); q.complete(token)
        except Exception:
            log.exception("Job failed")
            try: q.retry(token)
            except Exception: log.exception("Queue retry failed")
            time.sleep(5)

if __name__=="__main__": main()
