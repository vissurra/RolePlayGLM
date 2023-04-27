# RolePlayGLM

> 基于ChatGLM-6B使用ptuning进行微调，实现类instruction的效果

## 介绍
由于[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)不支持instruction，所以在进行角色扮演任务时具有不稳定性。本项目通过模板生成超小预料数据集，使用[ptuning](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning)进行微调，实现角色扮演的效果。
- Python 3.10

## 使用

1. 安装依赖
    ```shell
    # 安装ChatGLM-6B所需依赖
    $ pip install -r ChatGLM-6B/requirements.txt
    # 安装本项目所需依赖
    $ pip install -r requirements.txt
    ```

2. 构造数据集
    ```shell
    $ python dataset.py
    ```