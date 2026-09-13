"""
This runs a single experiment specified by the config file. A config file specifies:
- The specific model to use (e.g., llama-3.1-timex is a model that uses Llama 3.1 via OpenRouter for time expression extraction)
- The dataset to use (e.g., time expression extraction, spatial expression extraction)
- The evaluation method to use.
All of these are registered in the registry and can be specified by name in the config file.
"""

import argparse
from typing import Dict, List, Optional
import os, json, hashlib
from yaml import safe_load
from dotenv import load_dotenv


from mdg.registry import MODEL_WRAPPER_REGISTRY, DATASET_LOADER_REGISTRY, EVAL_METHOD_REGISTRY
from mdg.dataset_loaders import *
from mdg.evaluators import *
from mdg.methods import *
#import mdg.methods.few_shot.openrouter_based

def main():
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    parser = argparse.ArgumentParser(description="Run an experiment specified configurations.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output results.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = safe_load(f)
    print(f"Loaded configuration from {args.config}")

    method_config = config.get("method", {})
    dataset_config = config.get("dataset", {})
    evaluator_configs = config.get("evaluators", [])

    dataset_loader = DATASET_LOADER_REGISTRY[dataset_config["name"]](config=dataset_config.get("init_params", {}))
    method = MODEL_WRAPPER_REGISTRY[method_config["name"]](config=method_config.get("init_params", {}))
    evaluators = [
        EVAL_METHOD_REGISTRY[eval_config["name"]](config=eval_config.get("init_params", {}))
        for eval_config in evaluator_configs
    ]


    data: Dict[str, Optional[List]] = dataset_loader.run(**dataset_config.get("run_params", {}))
    try:
        assert all(v is not None for v in data.values()), "Dataset loader returned None data"
    except AssertionError as e:
        print(e)
        return
    print(f"Loaded data:", [f"{k}: {len(v) if v is not None else 0}" for k, v in data.items()])
    test_data = data.get("test", [])

    max_docs = method_config.get("run_params", {}).get("max_docs")
    if max_docs is not None:
        test_data = test_data[:int(max_docs)]

    predictions = method.run(
    test_data=test_data,
    train_data=data.get("train"),
    #dev_data=data.get("validation") or data.get("dev"),
    **method_config.get("run_params", {}),
    )
    print(f"Generated {len(predictions)} predictions.")

    if len(predictions) != len(test_data):
         print(f"[WARN] predictions len {len(predictions)} != test_data len {len(test_data)}")

    missing_docid = sum(1 for d in predictions if getattr(d, "doc_id", None) is None)
    types = {type(d).__name__ for d in predictions[:5]}
    #print(f"preds: n={len(predictions)}, missing_doc_id={missing_docid}, sample_types={types}")

    gold_ids = {getattr(d, "doc_id", None) for d in test_data}
    pred_ids = {getattr(d, "doc_id", None) for d in predictions}
    #print(f"doc_id overlap={len(gold_ids & pred_ids)} / {len(gold_ids)}")

    results = [
        evaluator.run(gold_data=test_data, predicted_data=predictions, **evaluator_config.get("run_params", {}))
        for evaluator, evaluator_config in zip(evaluators, evaluator_configs)
    ]
    

    if args.output:
        # Ensure parent directory exists
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Stable hash of the canonicalized config
        cfg_json = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        config_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

        # Make results JSON-serializable (handles Pydantic v1/v2 or plain dicts)
        def to_dict(obj):
            if obj is None:
                return None
            for attr in ("model_dump", "dict"):  # pydantic v2 / v1
                fn = getattr(obj, attr, None)
                if callable(fn):
                    try:
                        return fn()
                    except TypeError:
                        pass
            return obj  # leave plain dicts/lists as-is (your grouped evaluator returns a dict)

        serializable_results = [to_dict(r) for r in results]

        payload = {
            "config_hash": config_hash,
            "config": json.loads(cfg_json),
            "results": serializable_results,  # will include your grouped {"dataset_name", "groups"} dicts
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
