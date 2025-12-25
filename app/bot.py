import streamlit as st
from helpers.utils import write_message
from agent import generate_response

# Page Config
st.set_page_config("chatbot juridique", page_icon=":robot_face:")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "bonjour! Je suis votre assistant juridique spécialisé en droit du travail marocain. Comment puis-je vous aider aujourd'hui?"},
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent
        response = generate_response(message)
        write_message('assistant', response)
        


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)