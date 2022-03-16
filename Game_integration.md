# Gym-retro game integration guide

This guide presents how to use the custom integration of games present in the Neuromod datasets.

## 1. Install gym-retro

The [gym-retro](https://retro.readthedocs.io/en/latest/) package is used to emulate the game, you can install it with pip :

`pip install gym-retro`

## 2. Import the rom
Because of copyright restrictions, the rom of the games (i.e. the files containing the game data) cannot be shared with the data. You thus have to find them on your own.

To import the rom, run the following command (with the path being the path to the directory containing the rom and not the path to the rom itself):

`python3 -m retro.import /path/to/your/ROMs/directory/`

>Note:
To make sure that the rom you have corresponds to the one used by Gym-retro, when importing the rom its SHA1 hash will be compared to the one in the rom.sha file. If the hash is different, Gym-retro won't import the game.

## 3. Use the custom Integration

Now that the rom is imported, you can use Gym-retro's emulator to generate replays or new runs of the game. However to be able to use the same states, scenarios and extract the same variables from the game, you have to use the same game integration. This is done by specifying the path to **parent folder** of the folder containing the integration (i.e. the parent of the folder containing the data.json, metadata.json, <scenario_name>.state and scenario_<scenario_name>.json files) which is the stimuli folder of the dataset. Note that the path must be **absolute** and not relative. Then when instanciating the emulator you have to scpecify that you will be using a custom integration by setting the parameter `inttype` as `retro.data.Integrations.CUSTOM_ONLY`.

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
