# llms/utils.py

def parser(llm_output: str, task: str) -> str:

    if task == "predict":
        index = llm_output.lower().find("the agent's final state is: ")
        if index != -1:
            return llm_output[index + len("The agent's final state is: "):].strip()
        else:
            return "" 

    if task == "plan":
        index = llm_output.lower().find("the llm's action sequence is: ")
        if index != -1:
            return llm_output[index + len("The LLM's action sequence is: "):].strip()
        else:
            return "" 
        
    if task == "decompose":
        # print(llm_output)
        start = llm_output.lower().find("<start>\n")
        end = llm_output.lower().find("\n<end>")
        # print(start,end)
        if start != -1:
            return llm_output[start + len("<start>\n"):end].strip()
        else:
            return ""
    return ""
