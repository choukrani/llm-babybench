# processors/__init__.py

# from processors.normal_babyai_bot_processor import NormalBabyAIBotProcessor
from processors.omniscient_babyai_bot_processor import OmniscientBabyAIBotProcessor

def get_processor(name: str):
    # if name == "normal_babyai_bot":
        # return NormalBabyAIBotProcessor()
    if name == "omniscient_babyai_bot":
        return OmniscientBabyAIBotProcessor()
    else:
        raise ValueError(f"Unknown processor: {name}")
    