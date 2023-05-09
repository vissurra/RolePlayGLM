import gradio as gr

from knowledge.base import dump_knowledge, list_knowledge, load_knowledge


def show_content(knowledge):
    if knowledge == 'New':
        visible = True
        content = ''
    else:
        visible = False
        content = load_knowledge(knowledge)
    return gr.update(value='', visible=visible), content


def upload_knowledge(knowledge, knowledge_new, knowledge_list, file):
    if len(knowledge_new) > 0:
        knowledge = knowledge_new
        knowledge_list.append(knowledge_new)

    with open(file.name, mode='r', encoding='utf-8') as file:
        content = file.read()
    dump_knowledge(knowledge, content)
    return knowledge, gr.update(value='', visible=False), knowledge_list, gr.update(value=None), content


def ui():
    knowledge_list = list_knowledge()
    knowledge_list = gr.State(knowledge_list)

    with gr.Row():
        with gr.Column(scale=4):
            knowledge = gr.Dropdown(knowledge_list.value, value=knowledge_list.value[0])
            knowledge_new = gr.Textbox('', placeholder='New knowledge name', show_label=False)
            file = gr.File(file_types=['.txt'], show_label=False)
            upload_btn = gr.Button('Upload')
        with gr.Column(scale=10):
            content = gr.Textbox('', label='Content', lines=40)

    knowledge.change(show_content, [knowledge], [knowledge_new, content])

    upload_btn.click(upload_knowledge,
                     inputs=[knowledge, knowledge_new, knowledge_list, file],
                     outputs=[knowledge, knowledge_new, knowledge_list, file, content])
