import os
import glob
import retro
import numpy as np
from random import random
from bids_loader.stimuli.game import replay_bk2


def test_replay_bk2(
    tmpdir,
    game="Airstriker-Genesis",
    skip_first_step=True,
    scenario=None,
    integration_path="tests/test_stimuli/dummy_custom_integration",
    inttype=retro.data.Integrations.CUSTOM_ONLY,
):
    integration_path = os.path.abspath(integration_path)
    retro.data.Integrations.add_custom_path(integration_path)
    emulator = retro.make(game, record=tmpdir, inttype=inttype, scenario=scenario)
    emulator.reset()

    list_frames = []
    list_rewards = []
    list_info = []
    list_keys = []
    list_dones = []
    list_audio = []
    list_audio_rate = []
    done = False
    i = 0

    while not done and i < 10000:
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
        list_audio.append(emulator.em.get_audio())
        list_audio_rate.append(emulator.em.get_audio_rate())
        list_keys.append(key)
        list_frames.append(obs)
        list_rewards.append(rew)
        list_info.append(info)
        list_dones.append(done)
        i += 1

    assert done, "Game not done after 10,000 steps."
    emulator.close()
    del emulator
    bk2_path = glob.glob(os.path.join(tmpdir, "*.bk2"))[0]

    for i, (frame, key, annotations, sound) in enumerate(
        replay_bk2(bk2_path, skip_first_step, scenario, inttype)
    ):
        assert np.array_equal(frame, list_frames[i]), "Replayed frame doesn't match."
        assert key == list_keys[i], print(
            "Replayed keypress doesn't match : replayed is ", key, "but recorded is ", list_keys[i]
        )
        assert annotations["reward"] == list_rewards[i], "Replayed reward doesn't match."
        assert annotations["done"] == list_dones[i], "Replayed done condition doesn't match."
        assert annotations["info"] == list_info[i], "Replayed info doesn't match."
        assert np.array_equal(sound["audio"], list_audio[i]), "Replayed audio doesn't match."
        assert np.array_equal(
            sound["audio_rate"], list_audio_rate[i]
        ), "Replayed audio rate doesn't match."
