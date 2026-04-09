"""
run_task_xarm6.py - Data collection script for xArm6 tasks in SAPIEN.

Equivalent to script/run_task.py but adapted for single-arm xArm6 tasks.
Key differences from run_task.py:
  - No wrist_camera_type config (single external camera setup)
  - Single-arm camera config saving (head + front/wrist, no left/right)
  - Uses D435 camera defaults for both head and front (wrist) cameras
  - No test_render dependency
"""

import sys
sys.path.append('./')

import sapien.core as sapien
from collections import OrderedDict
from envs import *
import importlib
import yaml
import json
import traceback
import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    """Dynamically import and instantiate a task class from envs/"""
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except AttributeError:
        raise SystemExit(f"No such task: {task_name}")
    return env_instance


def get_camera_config(camera_type):
    """Load camera config (fovy, w, h) from _camera_config.yml"""
    camera_config_path = os.path.join(parent_directory, '../task_config/_camera_config.yml')
    assert os.path.isfile(camera_config_path), "Camera config file is missing"

    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f'Camera type "{camera_type}" is not defined in _camera_config.yml'
    return args[camera_type]


def main():
    """
    Entry point for xArm6 data collection.
    Usage: echo "pick_and_place" | python script/run_task_xarm6.py
    """
    task_name = input("Enter task name: ") if sys.stdin.isatty() else input()

    task = class_decorator(task_name)
    task_config_path = f'./task_config/{task_name}.yml'

    assert os.path.isfile(task_config_path), f"Task config file missing: {task_config_path}"

    with open(task_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # ============= Head Camera (External Third-Person) =============
    head_camera_config = get_camera_config(args['head_camera_type'])
    args['head_camera_fovy'] = head_camera_config['fovy']
    args['head_camera_w'] = head_camera_config['w']
    args['head_camera_h'] = head_camera_config['h']

    # ============= Front Camera (Wrist-Mounted) =============
    front_camera_config = get_camera_config(args['front_camera_type'])
    args['front_camera_fovy'] = front_camera_config['fovy']
    args['front_camera_w'] = front_camera_config['w']
    args['front_camera_h'] = front_camera_config['h']

    # Print camera config
    print('============= Camera Config =============\n')
    print(f'Head Camera (External):\n'
          f'    type: {args["head_camera_type"]}\n'
          f'    fovy: {args["head_camera_fovy"]}\n'
          f'    w: {args["head_camera_w"]}\n'
          f'    h: {args["head_camera_h"]}')
    print(f'Front Camera (Wrist):\n'
          f'    type: {args["front_camera_type"]}\n'
          f'    fovy: {args["front_camera_fovy"]}\n'
          f'    w: {args["front_camera_w"]}\n'
          f'    h: {args["front_camera_h"]}')
    print('\n=======================================')

    args['save_path'] += '/' + str(args['task_name']) + '_' + str(args['head_camera_type'])
    run(task, args)


def run(Demo_class, args):
    """
    Main data collection loop.
    Phase 1: Find successful seeds (simulation without saving)
    Phase 2: Replay with data saving enabled
    """
    epid = 0
    seed_list = []
    suc_num = 0
    fail_num = 0
    print(f"\nTask: {args['task_name']}")
    print(f"Target episodes: {args['episode_num']}")
    print(f"Dual arm: {args.get('dual_arm', False)}")
    print('=' * 50)

    # ============= Phase 1: Find successful seeds =============
    if not args['use_seed']:
        print('\n[Phase 1] Finding successful seeds...\n')
        while suc_num < args['episode_num']:
            try:
                Demo_class.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                Demo_class.play_once()

                if Demo_class.plan_success and Demo_class.check_success():
                    print(f"  Episode {suc_num} SUCCESS (seed={epid})")
                    seed_list.append(epid)
                    suc_num += 1
                else:
                    print(f"  Episode {suc_num} FAIL   (seed={epid})")
                    fail_num += 1

                Demo_class.close()
                if args['render_freq']:
                    Demo_class.viewer.close()
                epid += 1

            except Exception as e:
                stack_trace = traceback.format_exc()
                print(f"  Episode {suc_num} ERROR  (seed={epid})")
                print(f"    {stack_trace}")
                fail_num += 1
                Demo_class.close()
                if args['render_freq']:
                    Demo_class.viewer.close()
                epid += 1

        # Save seeds
        seeds_dir = './task_config/seeds/'
        os.makedirs(seeds_dir, exist_ok=True)
        with open(os.path.join(seeds_dir, f'{args["task_name"]}.txt'), 'w') as f:
            for sed in seed_list:
                f.write(f"{sed} ")
        print(f'\n[Phase 1] Complete. Failed {fail_num} times.')

    else:
        print('\n[Phase 1] Using saved seeds...')
        with open(f'./task_config/seeds/{args["task_name"]}.txt', 'r') as f:
            seed_list = [int(i) for i in f.read().split()]

    # ============= Phase 2: Collect data using successful seeds =============
    if args['collect_data']:
        print(f'\n[Phase 2] Collecting data ({len(seed_list)} episodes)...\n')

        args['render_freq'] = 0
        args['is_save'] = True

        data_base_dir = args['save_path']

        for id in range(args['st_episode'], args['episode_num']):
            print(f'  Collecting episode {id}/{args["episode_num"]-1} (seed={seed_list[id]})...')

            Demo_class.setup_demo(now_ep_num=id, seed=seed_list[id], **args)

            # Scene info JSON
            info_dir = f'/home/paddy/rrc/1cross/General-Policy/dataset/RoboTwin/data/{args["task_name"]}_{args["head_camera_type"]}_pkl'
            os.makedirs(info_dir, exist_ok=True)
            info_file_path = os.path.join(info_dir, 'scene_info.json')

            if not os.path.exists(info_file_path):
                with open(info_file_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False)

            with open(info_file_path, 'r', encoding='utf-8') as f:
                info_db = json.load(f)

            info = Demo_class.play_once()
            info_db[f'{id}'] = info

            with open(info_file_path, 'w', encoding='utf-8') as f:
                json.dump(info_db, f, ensure_ascii=False)

            # Save camera configs alongside raw data
            if Demo_class.save_type.get('raw_data', True):
                head_config = Demo_class.get_camera_config(Demo_class.head_camera)
                front_config = Demo_class.get_camera_config(Demo_class.front_camera)

                # Head camera (external)
                for key in ["h_color", "h_depth", "h_pcd"]:
                    if key in Demo_class.file_path:
                        save_json(Demo_class.file_path[key] + "config.json", head_config)

                # Front camera (wrist)
                for key in ["f_color", "f_depth", "f_pcd"]:
                    if key in Demo_class.file_path:
                        save_json(Demo_class.file_path[key] + "config.json", front_config)

            Demo_class.close()
            print(f'    Done.')

        print(f'\n[Phase 2] Data collection complete!')
        print(f'  Saved to: {data_base_dir}')


if __name__ == "__main__":
    main()
