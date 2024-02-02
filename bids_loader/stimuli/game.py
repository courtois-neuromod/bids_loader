import retro
from retro.scripts.playback_movie import playback_movie


def replay_bk2(
    bk2_path,
    skip_first_step=True,
    game=None,
    scenario=None,
    inttype=retro.data.Integrations.CUSTOM_ONLY,
):
    """Make an iterator that replays a bk2 file, returning frames, keypresses and
    annotations.
    If using a custom game, the absolute path to the parent folder of
    the gameintegration folder should be added to retro :
    `retro.data.Integrations.add_custom_path(integration_path)`
    otherwise the game won't be found.

    Example
    -------
    ```
    all_frames = []
    all_keys = []
    for frame, keys, annotations, sound in replay_bk2(path):
        all_frames.append(frame)
        all_keys.append(keys)
    ```

    Parameters
    ----------
    bk2_path : str
        Path to the bk2 file to replay.
    skip_first_step : bool
        Whether to skip the first step before starting the replay. The intended use of
        gym retro is to do so (i.e. True) but if the recording was not initiated as
        intended per gym-retro, not skipping (i.e. False) might be required.
        Default is True.
    scenario : str
        Path to the scenario json file. If None, the scenario.json file in the game
        integration folder will be used. Default is None.
    inttype : retro Integration
        Type of retro integration to use. Default is
        `retro.data.Integrations.CUSTOM_ONLY` for custom integrations, for default
        integrations shipped with retro, use `retro.data.Integrations.STABLE`.

    Yields
    -------
    frame : numpy.ndarray
        Current frame of the replay, of shape (H,W,3).
    keys : list of bool
        Current keypresses, list of booleans stating whicn key is pressed or not. The
        ordered name of the keys is in `emulator.buttons`.
    annotations : dict
        Dictonary containing the annotations of the game : reward, done condition and
        the values of the variables that are extracted from the emulator's memory.
    sound : dict
        Dictionnary containing the sound output from the game : audio and audio_rate.
    """
    movie = retro.Movie(bk2_path)
    if game is None:
        game = movie.get_game()
    emulator = retro.make(game, scenario=scenario, inttype=inttype, render_mode=None)
    emulator.initial_state = movie.get_state()
    emulator.reset()
    if skip_first_step:
        movie.step()
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(emulator.num_buttons):
                keys.append(movie.get_key(i, p))
        frame, rew, term, trunc, info = emulator.step(keys)
        done = term or trunc
        sound = {
            "audio": emulator.em.get_audio(),
            "audio_rate": emulator.em.get_audio_rate(),
        }
        annotations = {"reward": rew, "done": done, "info": info}
        yield frame, keys, annotations, sound


def save_replay_movie(
    bk2_path,
    video_path,
    annotations_path,
    skip_first_step=True,
    inttype=retro.data.Integrations.CUSTOM_ONLY,
):
    """Generate a MP4 movie and an annotation numpy file from a bk2 file.

    Parameters
    ----------
    bk2_path : str
        Path to the bk2 file to replay.
    video_path: str
        Path to the output video file.
    annotations_path: str
        Path to the output annotations file, a numpy file containing the actions and
        info for each frame.
    skip_first_step : bool
        Whether to skip the first step before starting the replay. The intended use of
        retro is to do so (i.e. True) but if the recording was not initiated as
        intended per retro, not skipping (i.e. False) might be required.
        Default is True.
    inttype : retro Integration
        Type of retro integration to use. Default is
        `retro.data.Integrations.CUSTOM_ONLY` for custom integrations, for default
        integrations shipped with retro, use `retro.data.Integrations.STABLE`.
    """
    if video_path[-4:] != ".mp4":
        video_path += ".mp4"
    movie = retro.Movie(bk2_path)
    if skip_first_step:
        movie.step()
    emulator = retro.make(
        movie.get_game(), inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=None
    )
    emulator.initial_state = movie.get_state()
    emulator.reset()
    playback_movie(
        emulator,
        movie,
        video_file=video_path,
        npy_file=annotations_path,
        lossless="mp4",
    )
