# runner/components.py

from formatters import get_formatter
from processors import get_processor
from prompters import get_prompter
from evaluators import get_evaluator
from llms import get_llm

def load_components(config: dict):
    """
    Given a parsed config, instantiate formatter, processor, prompter, evaluator, and LLM.
    """
    # Inputs
    formatter = get_formatter(config["inputs"]["formatter"])
    processor = get_processor(config["inputs"]["processor"])
    prompter = get_prompter(config["inputs"]["prompter"])

    # Evaluator
    evaluator = get_evaluator(config["eval"]["evaluator"])

    # LLM
    llm_config = config["llm"]
    llm = get_llm(
        name=llm_config["name"],
        model_name=llm_config["model"],
        **{k: v for k, v in llm_config.items() if k not in ["name", "model"]}
    )

    return {
        "formatter": formatter,
        "processor": processor,
        "prompter": prompter,
        "evaluator": evaluator,
        "llm": llm,
    }
