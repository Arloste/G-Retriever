from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM


load_model = {
    "llm": LLM,
    "inference_llm": LLM,
    "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    "7b": "unsloth/llama-2-7b",
    "7b_chat": "unsloth/llama-2-7b-chat",
    "13b": "unsloth/llama-2-13b",
    # "13b_chat": "unsloth/llama-2-13b-chat", # no such model
}
