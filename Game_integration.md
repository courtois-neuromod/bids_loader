# Gym-retro game integration guide

This guide presents how to use the custom integration of games present in the Neuromod datasets.

## 1. Install gym-retro

The [gym-retro](https://retro.readthedocs.io/en/latest/) package is used to emulate the game, you can install it with pip :

`pip install gym-retro`

## 2. Add the rom to the custom integration folder

Because of copyright restrictions, the rom of the games (i.e. the files containing the game data) cannot be shared with the data. You thus have to find them on your own and then put them in the corresponding integration folder, which is `<dataset>/stimuli/<name of the game>`. The rom file should be named `rom.md`.

Fore example the path to the rom for the shinobi game should be : `shinobi/stimuli/ShinobiIIIReturnOfTheNinjaMaster-Genesis/rom.md`.

>Note:
To make sure that the rom you have corresponds to the one used in the integration, you should compare the SHA1 hash of the rom to the one in the rom.sha file in the integration folder.

## 3. Use the custom Integration

To use the custom integration in a script, you have to specify the path to the **parent folder** of the folder containing the integration (i.e. the parent of the folder containing the data.json, metadata.json, <scenario_name>.state and scenario_<scenario_name>.json files) which is the **stimuli** folder of the dataset. Note that the path must be **absolute** and not relative. Then when instantiating the emulator you have to specify that you will be using a custom integration by setting the parameter `inttype` as `retro.data.Integrations.CUSTOM_ONLY`.

Here is some code example to run the emulator for the game shinobi.

```
import os
import retro

integration_path = os.path.abspath('data/shinobi/stimuli')
retro.data.Integrations.add_custom_path(integration_path)
emulator = retro.make(
              "ShinobiIIIReturnOfTheNinjaMaster-Genesis",
              inttype=retro.data.Integrations.CUSTOM_ONLY,
              scenario="Level1-0",
            )
```
