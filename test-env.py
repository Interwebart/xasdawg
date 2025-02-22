#!/usr/bin/env python3
import unittest
import numpy as np
import time
from drone_env import DroneEnv
import rclpy

class TestDroneEnv(unittest.TestCase):
    """Test cases for DroneEnv"""
    
    def setUp(self):
        """Set up test environment"""
        self.env = DroneEnv()
    
    def tearDown(self):
        """Clean up after test"""
        if hasattr(self, 'env'):
            self.env.close()
            # Force ROS context cleanup
            if rclpy.ok():
                rclpy.shutdown()

    # Rest of the test cases remain the same...   
    def test_environment_initialization(self):
        """Test environment initialization"""
        print("\n=== Testing Environment Initialization ===")
        
        # Test action space
        self.assertEqual(self.env.action_space.shape[0], 1)
        self.assertEqual(self.env.action_space.low[0], -1.0)
        self.assertEqual(self.env.action_space.high[0], 1.0)
        print("✓ Action space verified")
        
        # Test observation space
        self.assertEqual(self.env.observation_space.shape[0], 3)
        np.testing.assert_array_equal(
            self.env.observation_space.low,
            np.array([0.0, -2.0, 0.0])
        )
        np.testing.assert_array_equal(
            self.env.observation_space.high,
            np.array([10.0, 2.0, 1.0])
        )
        print("✓ Observation space verified")
        
        # Test initial state
        self.assertEqual(self.env.target_height, 5.0)
        print("✓ Initial state verified")
    
    # def test_reset_function(self):
        # """Test environment reset functionality"""
        # print("\n=== Testing Reset Function ===")
        # 
        # Test default reset
        # observation, info = self.env.reset()
        # self.assertEqual(len(observation), 3)
        # self.assertIn('initial_height', info)
        # self.assertIn('is_armed', info)
        # self.assertIn('nav_state', info)
        # print("✓ Default reset verified")
        # 
        # Test reset with custom height
        # custom_height = 1.0
        # observation, info = self.env.reset(options={'initial_height': custom_height})
        # self.assertEqual(info['initial_height'], custom_height)
        # print("✓ Custom height reset verified")
        
        # Test reset timeout
        # with self.assertRaises(TimeoutError):
            # self.env.reset(options={'timeout': 0.1})
        # print("✓ Reset timeout test passed")
    
    def test_step_function(self):
        """Test environment step functionality"""
        print("\n=== Testing Step Function ===")
        
        self.env.reset()
        
        # Test various actions
        test_actions = [
            (np.array([0.0]), "hover"),
            (np.array([1.0]), "up"),
            (np.array([-1.0]), "down")
        ]
        
        for action, description in test_actions:
            print(f"\nTesting {description} action: {action}")
            
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Verify observation
            self.assertEqual(len(observation), 3)
            self.assertTrue(np.all(observation >= self.env.observation_space.low))
            self.assertTrue(np.all(observation <= self.env.observation_space.high))
            
            # Verify reward
            self.assertIsInstance(reward, (int, float))
            
            # Verify termination flags
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            
            # Give time to observe behavior
            time.sleep(0.1)
            
        print("✓ Step function verified")
    
    def test_reward_function(self):
        """Test reward calculation"""
        print("\n=== Testing Reward Function ===")
        
        # Test perfect scenario
        perfect_observation = np.array([5.0, 0.0, 1.0])
        perfect_reward = self.env._calculate_reward(perfect_observation)
        self.assertGreater(perfect_reward, 0)
        
        # Test far from target
        far_observation = np.array([0.0, 0.0, 1.0])
        far_reward = self.env._calculate_reward(far_observation)
        self.assertLess(far_reward, perfect_reward)
        
        # Test high velocity penalty
        high_velocity_observation = np.array([5.0, 1.5, 1.0])
        velocity_reward = self.env._calculate_reward(high_velocity_observation)
        self.assertLess(velocity_reward, perfect_reward)
        print("✓ Reward function verified")
    
    def test_termination_conditions(self):
        """Test termination conditions"""
        print("\n=== Testing Termination Conditions ===")
        
        test_cases = [
            (np.array([8.0, 0.0, 1.0]), True, "Too high"),
            (np.array([-1.0, 0.0, 1.0]), True, "Below ground"),
            (np.array([5.0, 2.5, 1.0]), True, "Too fast"),
            (np.array([5.0, 0.0, 1.0]), False, "Valid state")
        ]
        
        for observation, should_terminate, description in test_cases:
            terminated, _ = self.env._check_termination(observation)
            self.assertEqual(terminated, should_terminate, 
                           f"Failed termination test for {description}")
            print(f"✓ {description} termination test passed")
    
    def test_full_episode(self):
        """Test a complete episode"""
        print("\n=== Testing Full Episode ===")
        
        observation, info = self.env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100
        
        print("\nStarting episode...")
        while steps < max_steps:
            current_height = observation[0]
            action = np.array([0.5 if current_height < self.env.target_height else -0.5])
            
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                print(f"Episode ended after {steps} steps")
                break
            
            time.sleep(0.1)
        
        print(f"\nTotal steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print("✓ Full episode test completed")

if __name__ == '__main__':
    unittest.main()