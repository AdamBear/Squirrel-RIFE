import json

import click

@click.command()
@click.option('--frames', type=int, prompt="All Frames:", help='number of frames')
@click.option('--exp', type=int, prompt="exp:", help='duplicate count of each frame')
def generate_rpu_edit_json(frames: int, exp: int):
    duplicate_list = []
    for frame in range(frames):
        duplicate_list.append({'source': frame, 'offset': frame, 'length': 2 ** exp - 1})
    edit_dict = {'duplicate': duplicate_list}
    with open('rpu_duplicate_edit.json', 'w') as w:
        json.dump(edit_dict, w)


if __name__ == "__main__":
    generate_rpu_edit_json()
