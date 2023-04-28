from __future__ import annotations

import random
from dataclasses import dataclass, field

with open('data/character_names.txt', 'r') as f:
    character_names = [line.strip() for line in f.readlines()]

with open('data/character_favorites.txt', 'r') as f:
    character_favorites = [line.strip() for line in f.readlines()]

with open('data/creator_names.txt', 'r') as f:
    creator_names = [line.strip() for line in f.readlines()]


@dataclass
class Character:
    name: str
    gender: str
    birthday: str
    favorite: list[str] = field(default_factory=list)


@dataclass
class Someone:
    name: str


class Template:
    def __init__(self, name, gender, birthday):
        self.character = Character(name, gender, birthday)
        self.creator: Someone | None = None

    @staticmethod
    def random_init() -> Template:
        name = random.choice(character_names)
        gender = random.choice(['男', '女'])
        birthday = f'{random.randint(1, 12)}月{random.randint(1, 28)}日'

        template = Template(name, gender, birthday)
        template.character.favorite += random.sample(character_favorites, 3)
        template.creator = Someone(random.choice(creator_names))
        return template

    def get_character_qa(self) -> (str, str):
        rnd = random.randint(0, 4)
        match rnd:
            case 0:
                q = random.choice(['你是谁？', '你叫什么名字？', '你的名字是什么？'])
                a = f'我是{self.character.name}。'
            case 1:
                q = random.choice(['你是男生还是女生？', '你是男的还是女的？', '你是男的还是女的？'])
                a = f'我是{self.character.gender}的！'
            case 2:
                q = random.choice(['你的生日是什么时候？', '你是什么时候出生的'])
                a = f'我是{self.character.birthday}出生的。'
            case 3:
                q = random.choice(['你的兴趣爱好是什么？', '你喜欢做什么？', '你平时都在干什么？'])
                a = f'我喜欢{"、".join(self.character.favorite)}'
            case 4:
                q = random.choice(['你的创造者是谁？', '你的主人是谁？', '你是谁创造的？'])
                a = f'我的创造者是伟大的{self.creator.name}！'
            case _:
                raise ValueError('Invalid random number')

        return q, a

    def build_instruction_history(self, chat_username):
        with open('data/instruction.txt', 'r') as f:
            instruction_template = f.read()

        instruction = instruction_template.format(
            name=self.character.name,
            gender=self.character.gender,
            birthday=self.character.birthday,
            favorites='、'.join(self.character.favorite),
            creator=self.creator.name
        )
        instruction += f'\n现在你正在和一个名叫"{chat_username}"的人聊天。\n明白了你就回复"明白了。"'

        return [
            [instruction, '明白了。']
        ]
