import time
import numpy as np
from drone_env import (
    DroneEnv,
    MAX_POSITION_VALUE,
    MAX_VELOCITY_VALUE,
    DESIRED_HEIGHT
)

def test_drone_takeoff():
    env = None
    try:
        env = DroneEnv()
        observation, _ = env.reset()
        print("Дрон успешно инициализирован")
        
        target_z = DESIRED_HEIGHT
        target_action = np.array([
            0.0,
            0.0,
            target_z / MAX_POSITION_VALUE,
            0.0
        ], dtype=np.float32)
        
        print(f"\nЦелевая высота: {target_z} м")
        print("Начало взлета...")
        
        for step in range(500):
            observation, reward, done, _, _ = env.step(target_action)
            current_z = abs(observation[2])
            velocity_z = observation[5]
            
            print(f"Шаг {step+1}: Высота {current_z:.2f} м | "
                  f"Скорость: {velocity_z:.2f} м/с | "
                  f"Награда: {reward:.1f}")
            
            if abs(current_z - target_z) < 0.1:
                print(f"\nУспех: достигнута высота {current_z:.2f} м!")
                break
                
            if done:
                print("\nПреждевременное завершение!")
                break
                
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания!")
    except Exception as e:
        print(f"\nКритическая ошибка: {str(e)}")
    finally:
        if env:
            env.close()
        print("\nТест завершен")

if __name__ == "__main__":
    test_drone_takeoff()