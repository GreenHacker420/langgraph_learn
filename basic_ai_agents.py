from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="dolphin3:latest")

chat_history = ChatMessageHistory()

promt = PromptTemplate(
    input_variables=["user_input", "chat_history"],
    template="Previous conversation:\n{chat_history}\nUser: {user_input}\nOllama:",
)

def run_chain(user_input:str):
    chat_history_text = "\n".join([f"User: {msg.content}" if msg.type == "human" else f"Ollama: {msg.content}" for msg in chat_history.messages])
    prompt_text = promt.format(user_input=user_input, chat_history=chat_history_text)
    response = llm.invoke(prompt_text)
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response)
    return response

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    
    response = run_chain(user_input)
    print(f"Ollama: {response}")



# while True:
#     user_input = input("User: ")
#     if user_input.lower() == "exit":
#         break
    
#     response = llm.invoke(user_input)
#     print(f"Ollama: {response}")