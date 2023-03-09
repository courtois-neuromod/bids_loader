import retro


def replay_bk2(
    bk2_path, skip_first_step=True, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY
):
    """Make an iterator that replays a bk2 file, returning frames, keypresses and annotations.

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
        gym retro is to do so (i.e. True) but if the recording was not initiated as intended
        per gym-retro, not skipping (i.e. False) might be required. Default is True.
    scenario : str
        Path to the scenario json file. If None, the scenario.json file in the game integration
        folder will be used. Default is None.
    inttype : gym-retro Integration
        Type of gym-retro integration to use. Default is `retro.data.Integrations.CUSTOM_ONLY`
        for custom integrations, for default integrations shipped with gym-retro, use
        `retro.data.Integrations.STABLE`.

    Yields
    -------
    frame : numpy.ndarray
        Current frame of the replay, of shape (H,W,3).
    keys : list of bool
        Current keypresses, list of booleans stating whicn key is pressed or not. The ordered name
        of the keys is in `emulator.buttons`.
    annotations : dict
        Dictonary containing the annotations of the game : reward, done condition and the values of
        the variables that are extracted from the emulator's memory.
    sound : dict
        Dictionnary containing the sound output from the game : audio and audio_rate.
    """
    movie = retro.Movie(bk2_path)
    emulator = retro.make(movie.get_game(), scenario=scenario, inttype=inttype)
    emulator.initial_state = movie.get_state()
    emulator.reset()
    if skip_first_step:
        movie.step()
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(emulator.num_buttons):
                keys.append(movie.get_key(i, p))
        frame, rew, done, info = emulator.step(keys)
        sound = {"audio": emulator.em.get_audio(), "audio_rate": emulator.em.get_audio_rate()}
        annotations = {"reward": rew, "done": done, "info": info}
        yield frame, keys, annotations, sound


def get_variables_from_replay(bk2_fpath, skip_first_step, save_gif=False, duration=10, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY):

    # Replays bk2 and generate info structure (dict of lists)
    replay = replay_bk2(bk2_fpath, skip_first_step=skip_first_step, scenario=scenario, inttype=inttype)
    all_frames = []
    all_keys = []
    all_info = []
    for frame, keys, annotations, sound in replay:
        all_keys.append(keys)
        all_info.append(annotations["info"])
        if save_gif:
            all_frames.append(frame)
    repetition_variables = reformat_info(all_info, all_keys, bk2_fpath)
                                         
    if save_gif:
        all_frames = np.moveaxis(np.array(all_frames), -1, 1)
        save_GIF(all_frames, bk2_fpath.replace(".bk2", ".gif"), duration=duration, optimize=False)
    return repetition_variables

def reformat_info(info, keys, bk2_fpath):
    """
    Reformats the info structure for a dictionnary structure containing the relevant info.
    """
    repetition_variables = {}
    repetition_variables["filename"] = bk2_fpath
    repetition_variables["level"] = bk2_fpath.split("/")[-1].split("_")[-2]
    repetition_variables["subject"] = bk2_fpath.split("/")[-1].split("_")[0]
    repetition_variables["session"] = bk2_fpath.split("/")[-1].split("_")[1]
    repetition_variables["repetition"] = bk2_fpath.split("/")[-1].split("_")[-1].split(".")[0]
    movie = retro.Movie(bk2_fpath)
    emulator = retro.make(movie.get_game())
    emulator.initial_state = movie.get_state()
    repetition_variables["actions"] = emulator.buttons
    emulator.close()

    for key in info[0].keys():
        repetition_variables[key] = []
    for button in repetition_variables["actions"]:
        repetition_variables[button] = []
    
    for frame_idx, frame_info in enumerate(info):
        for key in frame_info.keys():
            repetition_variables[key].append(frame_info[key])
        for button_idx, button in enumerate(repetition_variables["actions"]):
            repetition_variables[button].append(keys[frame_idx][button_idx])
    
    return repetition_variables

def images_from_array(array):
    if isinstance(array, Tensor):
        array = array.numpy()
    mode = "P" if (array.shape[1] == 1 or len(array.shape) == 3) else "RGB"
    if array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    if mode == "RGB":
        array = np.moveaxis(array, 1, 3)
    if array.min() < 0 or array.max() < 1:  # if pixel values in [-0.5, 0.5]
        array = 255 * (array + 0.5)

    images = [Image.fromarray(np.uint8(arr), mode) for arr in array]
    return images
    
def save_GIF(array, path, duration=200, optimize=False):
    """Save a GIF from an array of shape (n_frames, channels, width, height),
    also accepts (n_frames, width, height) for grey levels.
    """
    assert path[-4:] == ".gif"
    images = images_from_array(array[0:-1:4])
    images[0].save(
        path, save_all=True, append_images=images[1:], optimize=optimize, loop=0, duration=duration)
