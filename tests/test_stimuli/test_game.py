import os
import retro
import numpy as np
from random import random
from bids_loader.stimuli.game import replay_bk2


def test_replay_bk2():
    bk2_path = "tests/test_stimuli/Airstriker-Genesis-Level1-000000.bk2"
    if os.path.exists(bk2_path):
        os.remove(bk2_path)
    emulator = retro.make(game="Airstriker-Genesis", record=os.path.dirname(bk2_path))
    emulator.reset()

    list_frames = []
    list_rewards = []
    list_info = []
    list_keys = []
    list_dones = []
    done = False

    while not done:
        # buttons are ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        # Buttons that don't do anything in the game are not saved in the bk2,
        # neither illegal combinations (e.g. LEFT+RIGHT)
        B_press = random() < 0.5
        A_press = random() < 0.5
        UP_press = random() < 0.5
        DOWN_press = not UP_press and random() < 0.5
        LEFT_press = random() < 0.5
        RIGHT_press = not LEFT_press and random() < 0.5
        key = [
            B_press,
            A_press,
            False,
            False,
            UP_press,
            DOWN_press,
            LEFT_press,
            RIGHT_press,
            False,
            False,
            False,
            False,
        ]

        obs, rew, done, info = emulator.step(key)
        list_keys.append(key)
        list_frames.append(obs)
        list_rewards.append(rew)
        list_info.append(info)
        list_dones.append(done)

    emulator.close()
    emulator = retro.make(game="Airstriker-Genesis")

    for i, (frame, key, annotations) in enumerate(replay_bk2(bk2_path, emulator)):
        assert np.array_equal(frame, list_frames[i]), print("Replayed frame doesn't match.")
        assert key == list_keys[i], print(
            "Replayed keypress doesn't match : replayed is ", key, "but recorded is ", list_keys[i]
        )
        assert annotations["reward"] == list_rewards[i], print("Replayed reward doesn't match.")
        assert annotations["done"] == list_dones[i], print("Replayed done condition doesn't match.")
        assert annotations["info"] == list_info[i], print("Replayed info doesn't match.")

    os.remove(bk2_path)
