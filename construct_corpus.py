import json
import sys
import fire
def read_jsonl(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield json.loads(line)
def write_jsonl(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')
def write_jsonl_append(filename, each_data):
    with open(filename, 'a+') as f:
        f.write(json.dumps(each_data) + '\n')

DEFAULT_KEYS = [
    'text',
    'id'
]

def main(
    inp_path: str,
    out_path: str,
    text_key: str = 'func_code_string'
    ):
    # read jsonl
    data = read_jsonl(inp_path)
    # write jsonl
    for i, each_data in enumerate(data):
        write_data = {
            'text': each_data[text_key],
            'id': i
        }
        write_jsonl_append(out_path, write_data)
if __name__ == '__main__':
    fire.Fire(main)

    

