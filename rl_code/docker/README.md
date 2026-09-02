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
```bash
tensorboard serve --logdir dqn_cartpole_tensorboard --host 0.0.0.0
```



----------------------------------
| rollout/            |          |
|    ep_len_mean      | 176      |
|    ep_rew_mean      | 176      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 1836     |
|    fps              | 1855     |
|    time_elapsed     | 52       |
|    total_timesteps  | 98018    |
| train/              |          |
|    learning_rate    | 0.0001   |
|    loss             | 0.00201  |
|    n_updates        | 24479    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 176      |
|    ep_rew_mean      | 176      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 1840     |
|    fps              | 1855     |
|    time_elapsed     | 53       |
|    total_timesteps  | 98654    |
| train/              |          |
|    learning_rate    | 0.0001   |
|    loss             | 0.148    |
|    n_updates        | 24638    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 175      |
|    ep_rew_mean      | 175      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 1844     |
|    fps              | 1855     |
|    time_elapsed     | 53       |
|    total_timesteps  | 99365    |
| train/              |          |
|    learning_rate    | 0.0001   |
|    loss             | 0.131    |
|    n_updates        | 24816    |
----------------------------------
