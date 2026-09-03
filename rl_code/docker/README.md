## Build the Docker Image

From this directory, build the image with:

```bash
docker build -t romrobotics/rl-env:cuda12_torch2_3_0 .
```

## Pull the Docker Image

```bash
docker pull romrobotics/rl-env:cuda12_torch2_3_0
```

## tensorboard
train နေတဲ့အချိန်လည်း ကြည့်လို့ရပါတယ်။
```bash
tensorboard serve --logdir dqn_cartpole_tensorboard --host 0.0.0.0
# or
tensorboard --logdir ./dqn_cartpole_tensorboard --host 0.0.0.0
```

