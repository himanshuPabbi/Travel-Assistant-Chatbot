import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Validate API Token
if not api_token:
    st.error("Hugging Face API token not found. Please set it in the .env file.")
    st.stop()

# Define model configuration
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
task = "text-generation"

# Initialize Hugging Face LLM once
llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=api_token,
    repo_id=repo_id,
    task=task
)

# App config
st.set_page_config(page_title="Yatra Sevak.AI", page_icon="🌍")
st.title("Yatra Sevak.AI ✈️")

# Define prompt template
template = """
You are a travel assistant chatbot named Yatra Sevak.AI. You help users plan their trips and provide travel-related information.

Chat history:
{chat_history}

User question:
{user_question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I am Yatra Sevak.AI. How can I assist you today?")]

# Function to get a response from the model
def get_response(user_query, chat_history):
    # Format chat history as text
    formatted_chat_history = "\n".join(
        f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in chat_history
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "chat_history": formatted_chat_history,
        "user_question": user_query,
    })

    # Clean unnecessary prefixes
    return response.replace("AI response:", "").replace("chat response:", "").replace("bot response:", "").strip()

# Display existing chat history
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query:
    # Append user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    # Get AI response
    response = get_response(user_query, st.session_state.chat_history)

    with st.chat_message("AI"):
        st.write(response)

    # Append AI response to history
    st.session_state.chat_history.append(AIMessage(content=response))
