# test_env.py
import numpy as np
from drone_env import DroneEnv
import time

def test_takeoff():
    # Создаем окружение
    env = DroneEnv()
    
    try:
        # Получаем начальное наблюдение
        observation, _ = env.reset()
        print("Начальное состояние:", observation)
        
        # Команда на взлет (пример действия)
        # [x, y, z, yaw] где z отрицательный для взлета в NED координатах
        takeoff_action = np.array([0.0, 0.0, -0.2, 0.0])  # Взлет на 2 метра
        
        # Выполняем 100 шагов для взлета и удержания позиции
        for step in range(100):
            observation, reward, done, truncated, info = env.step(takeoff_action)
            
            # Выводим информацию каждые 10 шагов
            if step % 10 == 0:
                print(f"Шаг {step}:")
                print(f"Позиция (x,y,z): {observation[:3]}")
                print(f"Награда: {reward}")
                
            if done:
                print("Эпизод завершен досрочно")
                break
                
            time.sleep(0.1)  # Небольшая задержка для наблюдения
            
    finally:
        env.close()
        print("Окружение закрыто")

if __name__ == "__main__":
    test_takeoff()