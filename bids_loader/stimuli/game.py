import retro


def replay_bk2(bk2_path, emulator):
    """Make an iterator that replays a bk2 file, returning frames, keypresses and annotations.

    Example
    -------
    ```
    all_frames = []
    all_keys = []
    for frame, keys, annotations in replay_bk2(path, emulator):
        all_frames.append(frame)
        all_keys.append(keys)
    ```

    Parameters
    ----------
    bk2_path : str
        Path to the bk2 file to replay.
    emulator : type
        Gym-retro emulator instance of the corresponding game with the corresponding custom
        integration (see TODO: make a tutorial about custom integration).

    Yields
    -------
    frame : numpy.ndarray
        Current frame of the replay, of shape (H,W,3).
    keys : list of bool
        Current keypresses, list of booleans stating whicn key is pressed or not. The ordered name
        of the keys is in `emulator.buttons`.
    annotations: dict
        Dictonary containing the annotations of the game : reward, done condition and the values of
        the variables that are extracted from the emulator's memory.
    """
    movie = retro.Movie(bk2_path)
    emulator.initial_state = movie.get_state()
    emulator.reset()
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(emulator.num_buttons):
                keys.append(movie.get_key(i, p))
        frame, rew, done, info = emulator.step(keys)
        annotations = {"reward": rew, "done": done, "info": info}
        yield frame, keys, annotations
