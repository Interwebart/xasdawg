#!/usr/bin/env python3
import gymnasium as gym
from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

# Импортируем нашу среду (или определяем в том же файле)
from px4_left_env import PX4LeftEnv

def main():
    # Создаём экземпляр среды
    env = PX4LeftEnv()

    # Можно обернуть в VecEnv, но для одной среды достаточно напрямую
    # env = make_vec_env(lambda: PX4LeftEnv(), n_envs=1)

    # Создаём модель PPO (можно SAC, TD3 и т.д.)
    model = PPO("MlpPolicy", env, verbose=1)

    # Обучаем 10_000 шагов
    model.learn(total_timesteps=10_000)

    # Сохраняем модель
    model.save("px4_left_model")

    # Тестируем
    obs, _ = env.reset()
    for i in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            print(f"Episode finished at step={i}, reward={reward}")
            break

    env.close()

if __name__ == "__main__":
    main()
