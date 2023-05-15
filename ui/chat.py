import gradio as gr
import mdtex2html
from loguru import logger

from knowledge.base import list_knowledge, load_knowledge

model = None
tokenizer = None


def postprocess(self, y):
    """Override Chatbot.postprocess"""
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def refresh_knowledge(knowledge):
    knowledge_list = list_knowledge()
    if knowledge is None and len(knowledge_list) > 0:
        knowledge = knowledge_list[0]
    return gr.update(choices=knowledge_list, value=knowledge)


def query_docs(knowledge_vs, history, user_input, topk=1):
    if knowledge_vs is None:
        return None
    query = ' '.join([user for user, bot in history[-1:]] + [user_input])
    logger.debug(f'Query: {query}')
    docs_and_scores = knowledge_vs.similarity_search_with_score(query, k=topk)

    docs = ''
    for doc, score in docs_and_scores:
        docs += doc.page_content
    return docs


def build_instruction(name, gender, birthday, creator, favorites, chat_to, doc, history):
    instruction = f'''你是一个生活在虚拟世界里的人工智能，以下是你的基本信息。
姓名:{name}
性别:{gender}
生日:{birthday}
兴趣爱好:{favorites}
创造者姓名:{creator}
现在你正在和一个名叫"{chat_to}"的人聊天。
明白了你就回复"明白了。'''
    if doc is not None:
        instruction += f'''\n###\n{doc}'''

    init_history = [
        [instruction, '明白了。']
    ]

    if len(history) == 0:
        history = init_history
    elif len(history) <= 3:
        history = init_history + history[1:]
    else:
        history = init_history + history[-3:]
    return history


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(user_input, chatbot, max_length, top_p, temperature, history):
    logger.debug(f'User input: {user_input}')
    chatbot.append((parse_text(user_input), ""))

    response = ''
    for response, history in model.stream_chat(tokenizer, user_input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(user_input), parse_text(response))

        yield chatbot, history
    logger.debug(f'Response: {response}')


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


def ui(input_model, input_tokenizer):
    global model, tokenizer
    model = input_model
    tokenizer = input_tokenizer

    knowledge_list = gr.State(list_knowledge())
    knowledge_default = knowledge_list.value[0] if len(knowledge_list.value) > 0 else None
    knowledge_vs = gr.State(load_knowledge(knowledge_default) if knowledge_default else None)
    related_doc = gr.State('')
    history = gr.State([])

    with gr.Row():
        with gr.Column(scale=4):
            name = gr.Textbox('Dio', label='Name', placeholder="Dio")
        with gr.Column(scale=4):
            gender = gr.Textbox('男', label='Gender', placeholder="男")
        with gr.Column(scale=4):
            birthday = gr.Textbox('1月1日', label='Birthday', placeholder="1月1日")
        with gr.Column(scale=4):
            creator = gr.Textbox('无敌的卖鱼强', label='Creator Name', placeholder="无敌的卖鱼强")
    favorites = gr.Textbox("吃面包、关在房间里、扛压路机、戳脑门", label='Favorites',
                           placeholder="吃面包、关在房间里、扛压路机、戳脑门")

    with gr.Row():
        with gr.Column(scale=10):
            chatbot = gr.Chatbot().style(height=700)
        with gr.Column(scale=4):
            # chat_to = gr.Textbox("无敌的卖鱼强", label='Chat To', placeholder="无敌的卖鱼强")
            with gr.Row():
                knowledge = gr.Dropdown(knowledge_list.value, value=knowledge_default, label='Knowledge')
                refresh_btn = gr.Button('Refresh')
            user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=7)
            submit_btn = gr.Button("Submit", variant="primary")
            empty_btn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    knowledge.change(load_knowledge, [knowledge], [knowledge_vs])
    refresh_btn.click(refresh_knowledge, [knowledge], [knowledge])
    submit_btn \
        .click(query_docs,
               [knowledge_vs, history, user_input],
               [related_doc]) \
        .then(build_instruction,
              [name, gender, birthday, creator, favorites, creator, related_doc, history],
              [history]) \
        .then(predict,
              [user_input, chatbot, max_length, top_p, temperature, history],
              [chatbot, history],
              show_progress=True) \
        .then(reset_user_input, [], [user_input])

    empty_btn.click(reset_state, outputs=[chatbot, history], show_progress=True)
