# rl-fighters

Prerequisites:
- [pipenv](https://pipenv.pypa.io/en/latest/)

Run to install python environment
```bash
pipenv install
```
Activate the pipenv environment
```bash
pipenv shell
```
Run to install gym environment (the game)
```bash
pip install -e gym-fighters
```

`python rl-agent/train.py` trains agents with tensorboard logs at `rl-agent/logs/gradient_tape`

Every 100 games a window will pop up with the game running in real time to see how they develop.
Once trained, watch them play running `python rl-agent/play.py`


Ubuntu requirements:

If you get error message `ImportError: Library "GLU" not found.` run `sudo apt-get install freeglut3-dev`
