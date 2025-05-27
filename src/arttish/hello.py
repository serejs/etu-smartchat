import os


def parse_file_name(file_name):
    return '.'.join(file_name.split("\\")[-1].split(".")[:-1])


print(parse_file_name(os.path.join(os.getcwd(), 'hello.py')))
