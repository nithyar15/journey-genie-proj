import argparse
import sys
import time
from pathlib import Path

import streamlit as st
from bot.client.client_settings import get_client, get_clients
from bot.client.llm_client import LlmClient
from bot.conversation.conversation_retrieval import ConversationRetrieval
from bot.memory.vector_memory import VectorMemory
from bot.model.model_settings import get_model_setting, get_models
from helpers.log import get_logger
from bot.memory.embedder import EmbedderHuggingFace
from helpers.prettier import prettify_source

from intent import predict_intent
from ner import prediction
from multiquery import gen_subqueries

logger = get_logger(__name__)


@st.cache_resource(experimental_allow_widgets=True)
def load_llm(llm_client: LlmClient, model_name: str, model_folder: Path) -> LlmClient:
    """
    Create a LLM session object that points to the model.
    """
    model_settings = get_model_setting(model_name)
    clients = [client.value for client in model_settings.clients]
    if llm_client not in clients:
        llm_client = clients[0]
    llm = get_client(llm_client, model_folder=model_folder, model_settings=model_settings)
    return llm


@st.cache_resource()
def load_conversational_retrieval(_llm: LlmClient) -> ConversationRetrieval:
    conversation_retrieval = ConversationRetrieval(_llm)
    return conversation_retrieval


def init_page(root_folder: Path) -> None:
    st.set_page_config(page_title="JourneyGenie", page_icon="💬", initial_sidebar_state="collapsed")

    left_column, central_column, right_column = st.columns([2, 1, 2])

    with left_column:
        st.write(" ")

    with central_column:
        st.image(str(root_folder / "images/bot-new.jpg"), use_column_width="auto")
        st.markdown("""<h4 style='text-align: center; color: grey;'></h4>""", unsafe_allow_html=True)

    with right_column:
        st.write(" ")

    st.sidebar.title("Options")


@st.cache_resource
def init_welcome_message() -> None:
    with st.chat_message("assistant"):
        st.write("How can JourneyGenie help you today?")


@st.cache_resource()
def load_index(vector_store_path: Path) -> VectorMemory:
    """
    Loads a Vector Memory index based on the specified vector store path.

    Args:
        vector_store_path (Path): The path to the vector store.

    Returns:
        VectorMemory: An instance of the VectorMemory class with the loaded index.
    """
    embedding = EmbedderHuggingFace().get_embedding()
    index = VectorMemory(vector_store_path=str(vector_store_path), embedding=embedding)

    return index



def init_chat_history(conversational_retrieval: ConversationRetrieval) -> None:
    """
    Initializes the chat history, allowing users to clear the conversation.
    """
    clear_button = st.sidebar.button("🗑️ Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
        conversational_retrieval.get_chat_history().clear()


def display_messages_from_history():
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def return_intents(intent):
    return intent

def return_entities(entity):
    return entity

def main(parameters) -> None:
    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    vector_store_path = root_folder / "vector_store" / "docs_index"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    client = parameters.client
    model = parameters.model
    max_new_tokens = parameters.max_new_tokens

    init_page(root_folder)
    llm = load_llm(client, model, model_folder)
    conversational_retrieval = load_conversational_retrieval(_llm=llm)
    index = load_index(vector_store_path)
    init_chat_history(conversational_retrieval)
    init_welcome_message()
    display_messages_from_history()

    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message in chat message container
        with st.chat_message("user"):
            gen_subqueries(user_input)
            intents = predict_intent(user_input)
            entities = prediction(user_input)
            return_intents(intents)
            return_entities(entities)
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner(
                text="Refining the question and Retrieving the docs – hang tight! "
            ):
                refined_user_input = conversational_retrieval.refine_question(user_input)
                retrieved_contents, sources = index.similarity_search(query=refined_user_input, k=4)
                if retrieved_contents:
                    full_response += "Here are the retrieved text chunks with a content preview: \n\n"
                    message_placeholder.markdown(full_response)

                    for source in sources:
                        full_response += prettify_source(source)
                        full_response += "\n\n"
                        message_placeholder.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    #full_response = 
                    full_response += "No relevant context found. \n\n"
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})


        # Display assistant response in chat message container
        start_time = time.time()
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for token in conversational_retrieval.answer(question=user_input, max_new_tokens=max_new_tokens):
                full_response += llm.parse_token(token)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        conversational_retrieval.update_chat_history(user_input, full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbot")

    client_list = get_clients()
    default_client = client_list[0]

    model_list = get_models()
    default_model = model_list[0]

    parser.add_argument(
        "--client",
        type=str,
        choices=client_list,
        help=f"Client to be used. Defaults to {default_client}.",
        required=False,
        const=default_client,
        nargs="?",
        default=default_client,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=model_list,
        help=f"Model to be used. Defaults to {default_model}.",
        required=False,
        const=default_model,
        nargs="?",
        default=default_model,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="The maximum number of tokens to generate in the answer. Defaults to 512.",
        required=False,
        default=512,
    )

    return parser.parse_args()


# streamlit run chatbot_app.py
if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)