def evaluate_candidate(metrics,min_map50=0.80,min_precision=0.80):
    for key in ("map50","precision"):
        if key not in metrics: return {"approved":False,"reason":f"Missing {key}"}
    if metrics["map50"]<min_map50: return {"approved":False,"reason":"mAP50 below threshold"}
    if metrics["precision"]<min_precision: return {"approved":False,"reason":"Precision below threshold"}
    return {"approved":True,"reason":"Configured thresholds passed"}
