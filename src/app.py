def get_answer(query: str) -> str:
    return query


if __name__ == '__main__':
    print('Type `!q` to exit')
    user_text = ''
    while user_text != '!q':
        user_text = input()
        print(get_answer(user_text))
