import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# လေ့ကျင့်မည့် Gymnasium/MuJoCo environment အမည်။ Ant-v4 တွင် observation နှင့်
# action နှစ်ခုစလုံးသည် continuous value များဖြစ်ပြီး PPO policy က continuous action ထုတ်ပေးသည်။
ENV_ID = "Ant-v4"

# Training အတွင်း environment အားလုံးက စုစုပေါင်းပြုလုပ်မည့် step အရေအတွက်။
# တန်ဖိုးကြီးလေလေ လေ့လာချိန်ကြာပြီး policy က ပိုလေ့လာနိုင်သော်လည်း computation ပိုလိုသည်။
TOTAL_TIMESTEPS = 1_000_000

# တစ်ချိန်တည်း parallel လုပ်မည့် environment အရေအတွက်။ Environment တစ်ခုချင်းစီကနေ
# state, action, reward, next state စသည့် experience များကို တစ်ပြိုင်နက် စုမည်။
# များလေလေ rollout data မြန်မြန်ရသော်လည်း GPU/CPU memory နှင့် system resource ပိုသုံးသည်။
NUM_ENVS = 4

# PPO update တစ်ကြိမ်မလုပ်မီ environment တစ်ခုစီမှ စုမည့် consecutive steps အရေအတွက်။
# ဒီလို စုထားသော observation, action, reward, done, value estimate နှင့် log probability
# များကို rollout data ဟုခေါ်သည်။ Update တစ်ကြိမ်တွင် rollout data အရွယ်အစားမှာ
# NUM_ENVS * NUM_STEPS = 4 * 2048 = 8192 transitions ဖြစ်သည်။
NUM_STEPS = 2048

# Rollout data ကို mini-batch အဖြစ်ခွဲပြီး gradient update လုပ်ရာတွင် mini-batch တစ်ခုစီ
# ပါဝင်မည့် sample အရေအတွက်။ သေးလေလေ update ပိုမကြာခဏဖြစ်သော်လည်း noise ပိုများနိုင်သည်။
BATCH_SIZE = 64

# Neural network weights ကို တစ်ကြိမ် update လုပ်ရာတွင် အသုံးပြုသည့် learning rate။
# ကြီးလွန်းလျှင် training မတည်ငြိမ်နိုင်ပြီး သေးလွန်းလျှင် learning နှေးနိုင်သည်။
LEARNING_RATE = 3e-4

# အနာဂတ် reward များကို လက်ရှိ return ထဲ ထည့်တွက်သည့် discount factor (0 မှ 1 ကြား)။
# 0.99 သည် အနာဂတ် reward ကို အလေးထားပြီး long-term behavior ကို လေ့လာစေသည်။
GAMMA = 0.99

# Generalized Advantage Estimation (GAE) တွင် အနာဂတ် TD errors များကို ဘယ်လောက်အထိ
# ဆက်လက်ထည့်မည်ကို သတ်မှတ်သည်။ 0.95 သည် bias နှင့် variance ကြား အသုံးများသော balance ဖြစ်သည်။
GAE_LAMBDA = 0.95

# Rollout တစ်ခုတည်းကို ပြန်အသုံးပြုပြီး policy/value network ကို gradient update လုပ်မည့်
# epoch အရေအတွက်။ များလွန်းလျှင် sample ကို overfit ဖြစ်နိုင်သည်။
UPDATE_EPOCHS = 10

# PPO clipped objective တွင် policy probability ratio ကို old policy နှင့် ဘယ်လောက်အထိ
# ပြောင်းခွင့်ပြုမည်ကို သတ်မှတ်သည်။ 0.2 ဆိုလျှင် ratio ကို အကြမ်းဖျင်း 0.8 မှ 1.2 အတွင်းကန့်သတ်သည်။
CLIP_RANGE = 0.2

# Policy entropy loss ၏ coefficient။ Exploration ကိုအားပေးရန် entropy ကိုထည့်သော်လည်း
# 0.0 ဖြစ်သောကြောင့် ဤ training တွင် entropy bonus သီးခြားမပေးထားပါ။
ENTROPY_COEF = 0.0

# Value function loss ကို PPO total loss ထဲ ပေါင်းထည့်ရာတွင် အသုံးပြုသည့် weight။
# Critic တန်ဖိုးခန့်မှန်းမှုကို policy learning နှင့် balance လုပ်ပေးသည်။
VALUE_LOSS_COEF = 0.5

# Gradient norm ကို ဒီတန်ဖိုးထက် မကျော်စေရန် clip လုပ်သည်။ Exploding gradients ကြောင့်
# training ပျက်သွားခြင်းကို ကာကွယ်ပြီး learning ကို ပိုတည်ငြိမ်စေသည်။
MAX_GRAD_NORM = 0.5

# Random number generators အတွက် seed။ တူညီသော environment/library version နှင့် run လျှင်
# ရလဒ်ကို ပြန်လည်စမ်းသပ်နိုင်ရန် အသုံးပြုသည်။
SEED = 1

# Training ပြီးနောက် PPO policy နှင့် value network ကို သိမ်းမည့် base path။
# Stable-Baselines3 က အလိုအလျောက် .zip extension ဖြင့် သိမ်းသည်။
MODEL_PATH = Path("rom_sb3_ppo_antv4")

# VecNormalize ၏ running observation/reward mean နှင့် standard deviation ကို သိမ်းမည့် path။
# Evaluation အချိန် training နှင့်တူသော observation normalization ကို ပြန်သုံးရန် လိုအပ်သည်။
VECNORMALIZE_PATH = Path("rom_sb3_ppo_antv4_vecnormalize.pkl")

# Stable-Baselines3 TensorBoard event files သိမ်းမည့် directory။
# `tensorboard --logdir sb3_ppo_antv4_tensorboard` ဖြင့် training metrics ကြည့်နိုင်သည်။
TENSORBOARD_LOG_DIR = Path("sb3_ppo_antv4_tensorboard")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train SB3 PPO on Ant-v4.")
	parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
	parser.add_argument("--clip-range", type=float, default=CLIP_RANGE)
	parser.add_argument("--n-envs", type=int, default=NUM_ENVS)
	parser.add_argument("--seed", type=int, default=SEED)
	return parser.parse_args()


def train(args: argparse.Namespace) -> None:
	# n_envs ခုကို parallel run လုပ်ပြီး rollout data ကို စုသည် (SB3 default: DummyVecEnv)။
	# Environment တစ်ခုက state ကိုကြည့်ပြီး action လုပ်ကာ reward နှင့် next state ရလာသော
	# transition များကို NUM_STEPS အထိ စုမည်။ ထို့နောက် PPO က ထို data ဖြင့် network update လုပ်သည်။
	env = make_vec_env(ENV_ID, n_envs=args.n_envs, seed=args.seed)
	# Ant-v4 လို high-dimensional continuous-control envs အတွက် running mean/std normalization က training stability ကို ကူညီတယ်
	env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
	model = PPO(
		"MlpPolicy", env, n_steps=args.num_steps, batch_size=BATCH_SIZE, n_epochs=UPDATE_EPOCHS,
		learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda,
		clip_range=args.clip_range, ent_coef=ENTROPY_COEF, vf_coef=VALUE_LOSS_COEF, max_grad_norm=MAX_GRAD_NORM,
		seed=args.seed, verbose=1, tensorboard_log=str(TENSORBOARD_LOG_DIR),
	)
	model.learn(total_timesteps=TOTAL_TIMESTEPS)
	model.save(MODEL_PATH)
	# testing/evaluation အချိန် observation ကို ပြန် normalize လုပ်ဖို့ VecNormalize stats ကိုပါ သိမ်း
	env.save(str(VECNORMALIZE_PATH))
	print(f"Training complete. Model saved to {MODEL_PATH}")
	env.close()


if __name__ == "__main__":
	train(parse_args())
