# RolePlayGLM

> 基于ChatGLM-6B使用ptuning进行微调，实现类instruction的效果

## 介绍

由于[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
不支持instruction，所以在进行角色扮演任务时具有不稳定性。本项目通过模板生成超小语料数据集，使用[ptuning](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning)
进行微调，实现角色扮演的效果。

- Python 3.10

### 示例

![demo](data/images/demo.png)

## 使用

1. 安装依赖
    ```shell
    # 安装ChatGLM-6B所需依赖
    $ pip install -r ChatGLM-6B/requirements.txt
    # 安装ptuning依赖
    $ pip install rouge_chinese nltk jieba datasets
    # 安装本项目所需依赖
    $ pip install -r requirements.txt
    ```

2. 构造数据集
    ```shell
    $ python dataset.py
    ```

3. 训练模型
   > 可以删除[train_instruction.sh](train_instruction.sh)中的```--quantization_bit```参数，不使用量化。
    ```shell
    $ bash train_instruction.sh
    ```

4. 运行
    ```shell
    $ bash web_demo.sh
    ```

5. 访问
    <br> 浏览器直接打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 硬件（训练）

| **量化**    | **显存(GB)** | **速度(s/it)** | **loss** |
|-----------|------------|--------------|----------|
| FP16（无量化） | 15         | 6.28         | 0.0021   |
| INT4      | 7          | 8.95         | 0.0012   |

- 以上数据为单张RTX 3090显卡的测试结果
- epoch: 50
