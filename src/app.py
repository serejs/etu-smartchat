from arttish.model import get_model


def get_answer(query: str) -> str:
    return get_model().invoke(query).content


if __name__ == '__main__':
    print('Type `!q` to exit')
    user_text = input()
    while user_text != '!q':
        print(get_answer(user_text))
        user_text = input()
