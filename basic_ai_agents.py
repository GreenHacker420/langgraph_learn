from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="dolphin3:latest")


while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    
    response = llm.invoke(user_input)
    print(f"Ollama: {response}")