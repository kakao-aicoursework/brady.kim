import os
import json
import openai
import tkinter as tk
import pandas as pd
from tkinter import scrolledtext
import chromadb
from chromadb.utils import embedding_functions
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate


client = chromadb.PersistentClient()
openai_ef = {}
collection = {}


def read_and_parse_file(file_path):
    data = []
    current_title = None
    current_content = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                if current_title is not None:
                    data.append({'title': current_title, 'content': '\n'.join(current_content)})
                    current_content = []
                current_title = line[1:].strip()
            else:
                current_content.append(line)

        if current_title is not None:
            data.append({'title': current_title, 'content': '\n'.join(current_content)})

    return pd.DataFrame(data)


def store_data_to_vector_db(df):
    ids = []
    documents = []

    for idx in range(len(df)):
        item = df.iloc[idx]
        id = item['title'].replace(' ','-')
        document = f"Title: {item['title']}, Content: {item['content']}"
        ids.append(id)
        documents.append(document)

    collection.add(
        documents=documents,
        ids=ids,
    )

def search_vector_db(query):
    query_texts = []
    if isinstance(query, str):
       query_texts.append(query)
    else:
        query_texts = query
    res = collection.query(query_texts=query_texts)
    return json.dumps(res, ensure_ascii=False)

def get_query(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=message_log,
            temperature=temperature,
            functions=functions,
            function_call='auto',
        )

        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            available_functions = {
                "search": search_vector_db,
            }
            function_name = response_message["function_call"]["name"]
            #fuction_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            return function_args
        return None


def send_message(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        functions=functions,
        function_call='auto',
    )

    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = {
            "search": search_vector_db,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
        # **function_args로 처리하기
        function_response = fuction_to_call(**function_args)

        # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
        message_log.append(response_message)  # GPT의 지난 답변을 message_logs에 추가하기
        message_log.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # 함수 실행 결과도 GPT messages에 추가하기
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=message_log,
            temperature=temperature,
        )  # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기
    return response.choices[0].message.content

def send_message_v2(query_args, message, gpt_model="gpt-3.5-turbo", temperature=0.1):
    print(query_args)
    if query_args is None:
        search_result=""
    else:
        search_result=search_vector_db(query=query_args['query'])

    llm = ChatOpenAI(temperature=temperature, max_tokens=200, model=gpt_model)

    prompt_template = ChatPromptTemplate.from_template(
        template='''
        <검색 결과>
        {search_result}
        </검색 결과>
        
        너는 카카오싱크에 대한 user의 질문에 답변해야해. <검색 결과>를 활용하여 카카오싱크에 대한 user의 질문에 대해 요약해서 답해줘.
        '''
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain(dict(
        search_result=search_result,
    ))

    print(result)

    return result['text']


def main():
    message_log = [
        {
            "role": "system",
            "content": '''
            assistant는 user가 하는 카카오싱크에 대한 질문에대해 답해라.
            '''
        }
    ]

    functions = [
        {
            "name": "search",
            "description": '카카오싱크에 대한 정보 검색',
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "카카오싱크에 대한 정보가 담긴 vector database에 질의할 검색어 in Korean",
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
        #response = send_message(message_log, functions)
        response = get_query(message_log, functions)
        response = send_message_v2(response, user_input)
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
    with open('../openai_api_key', 'r') as file:
        # Read the first line of the file
        first_line = file.readline().strip()
        os.environ["OPENAI_API_KEY"] = first_line
        openai.api_key = first_line

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai.api_key,
        model_name="text-embedding-ada-002"
    )

    collection = client.get_or_create_collection(
        name="kakaotalk_sync",
        embedding_function = openai_ef,
        metadata={"hnsw:space": "cosine"}  # l2 is the default
    )

    df = read_and_parse_file('project_data_카카오싱크.txt')
    store_data_to_vector_db(df)

    main()