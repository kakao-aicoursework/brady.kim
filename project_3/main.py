import os
import openai
import tkinter as tk
from tkinter import scrolledtext
from langchain.chains import ConversationChain, LLMChain, LLMRouterChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter

with open('../openai_api_key', 'r') as file:
    # Read the first line of the file
    first_line = file.readline().strip()
    openai.api_key = first_line
    os.environ["OPENAI_API_KEY"] = first_line

##################################################
def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template
##################################################
def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )
##################################################
SOCIAL_STEP1_PROMPT_TEMPLATE = "./prompt_templates/social_1.txt"
SYNC_STEP1_PROMPT_TEMPLATE = "./prompt_templates/sync_1.txt"
CHAN_STEP1_PROMPT_TEMPLATE = "./prompt_templates/chan_1.txt"
INTENT_PROMPT_TEMPLATE = "./prompt_templates/parse_intent.txt"
INTENT_LIST_TXT = "./prompt_templates/intent_list.txt"
##################################################
llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")

social_step1_chain = create_chain(
    llm=llm,
    template_path=SOCIAL_STEP1_PROMPT_TEMPLATE,
    output_key="social-step1",
)
sync_step1_chain = create_chain(
    llm=llm,
    template_path=SYNC_STEP1_PROMPT_TEMPLATE,
    output_key="sync-step1",
)
chan_step1_chain = create_chain(
    llm=llm,
    template_path=CHAN_STEP1_PROMPT_TEMPLATE,
    output_key="chan-step1",
)
parse_intent_chain = create_chain(
    llm=llm,
    template_path=INTENT_PROMPT_TEMPLATE,
    output_key="intent",
)
default_chain = ConversationChain(llm=llm, output_key="text")
##################################################
destinations = [
    "카카오소셜 질문: 카카오소셜 서비스에 대한 질문",
    "카카오싱크 질문: 카카오싱크 서비스에 대한 질문",
    "카카오톡채널 질문: 카카오톡채널 서비스에 대한 질문",
]
destinations = "\n".join(destinations)
##################################################
router_prompt_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations)
router_prompt = PromptTemplate.from_template(
    template=router_prompt_template, output_parser=RouterOutputParser()
)
router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt, verbose=True)
##################################################
multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains={
        "카카오소셜 질문": social_step1_chain,
        "카카오싱크 질문": sync_step1_chain,
        "카카오톡채널 질문": chan_step1_chain,
    },
    default_chain=ConversationChain(llm=llm, output_key="text"),
)
##################################################
_db_social = Chroma(
    persist_directory="./chroma-social",
    embedding_function=OpenAIEmbeddings(),
    collection_name="social-db",
)
_retriever_social = _db_social.as_retriever()
_db_sync = Chroma(
    persist_directory="./chroma-sync",
    embedding_function=OpenAIEmbeddings(),
    collection_name="sync-db",
)
_retriever_sync = _db_sync.as_retriever()
_db_chan = Chroma(
    persist_directory="./chroma-chan",
    embedding_function=OpenAIEmbeddings(),
    collection_name="chan-db",
)
_retriever_chan = _db_chan.as_retriever()
##################################################
def upload_embedding_from_file(file_path, collection_name, persist_dir):
    loader = UnstructuredMarkdownLoader
    documents = loader(file_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(docs, end='\n\n\n')

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    print('db success')
##################################################
upload_embedding_from_file("./project_data_카카오소셜.txt", "social-db", "./chroma-social/")
upload_embedding_from_file("./project_data_카카오싱크.txt", "sync-db", "./chroma-sync/")
upload_embedding_from_file("./project_data_카카오톡채널.txt", "chan-db", "./chroma-chan/")
##################################################
def query_db_social(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever_social.get_relevant_documents(query)
    else:
        docs = _db_social.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs
def query_db_sync(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever_sync.get_relevant_documents(query)
    else:
        docs = _db_sync.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs
def query_db_chan(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever_chan.get_relevant_documents(query)
    else:
        docs = _db_chan.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs
##################################################
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
HISTORY_DIR = "./chat_histories"
def load_conversation_history(conversation_id: str):
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    return FileChatMessageHistory(file_path)


def log_user_message(history: FileChatMessageHistory, user_message: str):
    history.add_user_message(user_message)


def log_bot_message(history: FileChatMessageHistory, bot_message: str):
    history.add_ai_message(bot_message)


def get_chat_history(conversation_id: str):
    history = load_conversation_history(conversation_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
    )

    return memory.buffer
##################################################
def generate_answer(user_message, conversation_id='fa1010') -> dict[str, str]:
    history_file = load_conversation_history(conversation_id)

    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)
    context["chat_history"] = get_chat_history(conversation_id)

    intent = parse_intent_chain.run(context)

    print("INTENT:", intent)

    if intent == "카카오소셜 질문":
        context["related_documents"] = query_db_social(context["user_message"])
        answer = social_step1_chain.run(context)
    elif intent == "카카오싱크 질문":
        context["related_documents"] = query_db_sync(context["user_message"])
        answer = sync_step1_chain.run(context)
    elif intent == "카카오톡채널 질문":
        context["related_documents"] = query_db_chan(context["user_message"])
        answer = chan_step1_chain.run(context)
    else:
        answer = default_chain.run(context["user_message"])

    log_user_message(history_file, user_message)
    log_bot_message(history_file, answer)

    return {"answer": answer}
##################################################

def send_message(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
    res = generate_answer(message_log)
    return res["answer"]

def main():
    message_log = [
        #{
        #    "role": "system",
        #    "content": '''
        #    You are an assistant to help users to understand features of the following services: 카카오소셜, 카카오싱크, and 카카오톡채널".
        #    Users can ask questions related to 카카오소셜, 카카오싱크, or 카카오톡채널, or engage in general conversation..
        #    Your user will be Korean, so communicate in Korean.
        #    '''
        #}
    ]

    functions = [
        {
            "name": "query_vector_db",
            "description": 'search the information about "카카오톡 채널."',
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "vector database에 질의할 쿼리문 in Korean",
                    },
                },
                "required": ["query"],
            },
        }
    ]

    def show_popup_message(window, message):
        popup = tk.Toplevel(window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = window.winfo_x()
        window_y = window.winfo_y()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(window)
        popup.attributes('-topmost', True)

        popup.update()
        return popup

    def on_send():
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            window.destroy()
            return

        message_log.append({"role": "user", "content": user_input})
        conversation.config(state=tk.NORMAL)  # 이동
        conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = show_popup_message(window, "처리중...")
        window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response = send_message(user_input, functions)
        #response = send_message(message_log, functions)
        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("GPT AI")

    font = ("맑은 고딕", 10)

    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    send_button = tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)

    window.bind('<Return>', lambda event: on_send())
    window.mainloop()

if __name__ == "__main__":
    main()