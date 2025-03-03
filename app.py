import gradio as gr
from flask import Flask, request, jsonify
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from typing import List
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain import hub
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Define the embedding model
class EmbeddingModel:
    def __init__(self, model):
        self.model = SentenceTransformer(model, trust_remote_code=True)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
            
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()

def custom_relevance_score_fn(similarity_score: float) -> float:
    # Example calculation (customize as needed)
    relevance_score = 1 / (1 + similarity_score)
    return relevance_score

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Define the state structure
class State(TypedDict):
    question: str
    messages: Annotated[list, add_messages]
    retrieved_content: str
    conversation_state: str

# RAG node: retrieves context and generates an answer.
def rag_node(state):
    embeddings_fn = EmbeddingModel('all-MiniLM-L6-v2')
    vectorstore = Chroma(
        persist_directory="conv_logs",
        embedding_function=embeddings_fn,
        relevance_score_fn=custom_relevance_score_fn
    )

    question = state["question"]
    docs = vectorstore.similarity_search_with_relevance_scores(query=question)
    retrieved_content = ''
    for doc in docs[:2]:
        content, score = doc[0].page_content, doc[1]
        retrieved_content += content + '\n'
    prompt = hub.pull("rlm/rag-prompt")
    # Adjust the prompt template to include the question and retrieved context.
    prompt.messages[0].prompt.template = (
        "You are an assistant for question-answering tasks. Use the retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. Keep the answer concise. Think step by step. \n"
        "Question: {question} \nContext: {context} \nAnswer:"
    )
    messages = prompt.invoke({"question": state["question"], "context": retrieved_content})
    print("Messages: ", messages)
    response = llm.invoke(messages)
    print('Response :', response)
    state["messages"] = [
        HumanMessage(content=question),
        AIMessage(content=response.content)
    ]
    state["retrieved_content"] = retrieved_content
    return state

# Classification agent: categorizes the conversation.
def classification_agent(state):
    # Combine all messages into a transcript.
    conversation = "\n".join([
        f"{msg.__class__.__name__}: {msg.content}" 
        for msg in state["messages"]
    ])

    # Create a prompt instructing the agent to classify the conversation.
    classification_prompt = (
        "You are a conversation classifier. Given the conversation transcript below, "
        "categorize the conversation into one of the following categories:\n"
        "  - Casual Chat\n"
        "  - Discussing Hobbies\n"
        "  - Discussing Work\n"
        "  - Discussing Personal Matters\n\n"
        "Conversation Transcript:\n"
        f"{conversation}\n\n"
        "Please respond with only the category name."
    )

    # Invoke the LLM with the classification prompt.
    response = llm.invoke([HumanMessage(content=classification_prompt)])

    # Save the classification result in the state.
    state["conversation_state"] = response.content.strip()
    return state

# Build the state graph.
graph_builder = StateGraph(State)
graph_builder.add_node("rag_node", rag_node)
graph_builder.add_node("agent", classification_agent)
graph_builder.add_edge(START, "rag_node")
graph_builder.add_edge("rag_node", "agent")
graph_builder.add_edge("agent", END)

app_graph = graph_builder.compile()

# Define the function that processes a single user input.
def run_agent(user_question: str, history: list):
    inputs = {"question": user_question}
    final_state = None
    # Stream through the state graph.
    for output in app_graph.stream(inputs, {"recursion_limit": 150}):
        final_state = output
    # Extract responses:
    rag_response = final_state['agent']['messages'][-1].content
    agent_response = final_state['agent']['conversation_state']
    # Append the new turn to the conversation history.
    history.append((f"User: {user_question}", f"RAG: {rag_response}\nAgent Classification: {agent_response}"))
    return history, ""

# Define the function that handles the exit action.
def exit_conversation(history):
    # When exit is clicked, clear the conversation and disable further input.
    history = [("**Conversation ended.**", "Thank you for chatting.")]
    return history, gr.update(interactive=False)

# Build a persistent conversation Blocks interface.
with gr.Blocks() as demo:
    gr.Markdown("## RAG & Conversation Classification Agent\n\nType your question below. The conversation will remain open until you click the **Exit** button.")
    
    chatbot = gr.Chatbot(label="Conversation History")
    state = gr.State([])  # to keep the conversation history
    
    with gr.Row():
        txt_input = gr.Textbox(label="Your Question", placeholder="Enter your question here...", interactive=True)
        exit_btn = gr.Button("Exit")
    
    # When a user submits a question, update the conversation.
    txt_input.submit(run_agent, inputs=[txt_input, state], outputs=[chatbot, txt_input])
    # When the exit button is clicked, clear the conversation and disable further inputs.
    exit_btn.click(exit_conversation, inputs=state, outputs=[chatbot, txt_input])

if __name__ == "__main__":
    demo.launch()
