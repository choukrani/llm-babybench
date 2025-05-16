# evaluators/__init__.py

from evaluators.predict import PredictEvaluator
from evaluators.plan import PlanEvaluator
from evaluators.decompose import DecomposeEvaluator

def get_evaluator(name: str):
    if name == "predict":
        return PredictEvaluator()
    elif name == "plan":
        return PlanEvaluator()
    elif name == "decompose":
        return DecomposeEvaluator()
    else:
        raise ValueError(f"Unknown evaluator: {name}")
    