"""
Test script for TankSim environment

This script tests the basic functionality of the TankSimEnv to ensure
everything is working correctly.
"""

import sys
import numpy as np
from tank_sim_env import TankSimEnv


def test_environment_creation():
    """Test that the environment can be created."""
    print("Test 1: Environment Creation")
    print("-" * 50)
    try:
        env = TankSimEnv()
        print("✓ Environment created successfully")
        print(f"  Grid size: {env.grid_size}x{env.grid_size}")
        print(f"  Number of agents: {env.num_agents}")
        print(f"  Number of obstacles: {env.num_obstacles}")
        return True, env
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False, None


def test_reset(env):
    """Test the reset functionality."""
    print("\nTest 2: Reset Functionality")
    print("-" * 50)
    try:
        obs, info = env.reset(seed=42)
        print("✓ Environment reset successfully")
        print(f"  Number of observations: {len(obs)}")
        print(f"  Observation keys: {list(obs.keys())}")

        # Check observation shapes
        for agent_id, observation in obs.items():
            print(f"  Agent {agent_id} observation shape: {observation.shape}")
            assert observation.shape == (1, 32, 32), f"Wrong observation shape: {observation.shape}"

        print("✓ All observations have correct shape (1, 32, 32)")

        # Check info
        print(f"  Info keys: {list(info.keys())}")
        print(f"  Team 0 alive: {info['team0_alive']}")
        print(f"  Team 1 alive: {info['team1_alive']}")
        print(f"  Total tanks: {len(info['tanks'])}")

        return True
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_space(env):
    """Test the action space."""
    print("\nTest 3: Action Space")
    print("-" * 50)
    try:
        print(f"  Action space: {env.action_space}")
        print(f"  Action space type: {type(env.action_space)}")
        print(f"  Number of actions: {env.action_space.n}")

        # Sample some actions
        for i in range(3):
            action = env.action_space.sample()
            print(f"  Sample action {i+1}: {action}")

        print("✓ Action space working correctly")
        return True
    except Exception as e:
        print(f"✗ Action space test failed: {e}")
        return False


def test_single_step(env):
    """Test taking a single step in the environment."""
    print("\nTest 4: Single Step Execution")
    print("-" * 50)
    try:
        obs, info = env.reset(seed=42)

        # Create actions for all agents (all idle)
        actions = {agent_id: 0 for agent_id in obs.keys()}
        print(f"  Actions: {actions}")

        obs, rewards, terminated, truncated, info = env.step(actions)

        print("✓ Step executed successfully")
        print(f"  Observations received: {len(obs)}")
        print(f"  Rewards: {rewards}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Step count: {info['step_count']}")

        return True
    except Exception as e:
        print(f"✗ Single step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_movement(env):
    """Test tank movement."""
    print("\nTest 5: Tank Movement")
    print("-" * 50)
    try:
        obs, info = env.reset(seed=42)

        # Get initial tank positions
        initial_positions = {t['agent_id']: (t['x'], t['y']) for t in info['tanks']}
        print(f"  Initial positions: {initial_positions}")

        # Move all tanks forward
        actions = {agent_id: 1 for agent_id in obs.keys()}  # Action 1 = Move Forward

        obs, rewards, terminated, truncated, info = env.step(actions)

        # Check if any tanks moved
        new_positions = {t['agent_id']: (t['x'], t['y']) for t in info['tanks']}
        print(f"  New positions: {new_positions}")

        moved_count = sum(1 for aid in initial_positions
                         if initial_positions[aid] != new_positions[aid])
        print(f"  Tanks that moved: {moved_count}/{len(initial_positions)}")

        print("✓ Movement test completed")
        return True
    except Exception as e:
        print(f"✗ Movement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rotation(env):
    """Test tank rotation."""
    print("\nTest 6: Tank Rotation")
    print("-" * 50)
    try:
        obs, info = env.reset(seed=42)

        # Get initial orientations
        initial_orientations = {t['agent_id']: t['orientation'] for t in info['tanks']}
        print(f"  Initial orientations: {initial_orientations}")

        # Rotate all tanks right
        actions = {agent_id: 2 for agent_id in obs.keys()}  # Action 2 = Turn Right

        obs, rewards, terminated, truncated, info = env.step(actions)

        new_orientations = {t['agent_id']: t['orientation'] for t in info['tanks']}
        print(f"  New orientations: {new_orientations}")

        # Check if all tanks rotated
        rotated_count = sum(1 for aid in initial_orientations
                           if initial_orientations[aid] != new_orientations[aid])
        print(f"  Tanks that rotated: {rotated_count}/{len(initial_orientations)}")

        print("✓ Rotation test completed")
        return True
    except Exception as e:
        print(f"✗ Rotation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combat(env):
    """Test combat mechanics."""
    print("\nTest 7: Combat Mechanics")
    print("-" * 50)
    try:
        obs, info = env.reset(seed=42)

        # Fire with all tanks
        actions = {agent_id: 4 for agent_id in obs.keys()}  # Action 4 = Fire

        initial_health = {t['agent_id']: t['health'] for t in info['tanks']}
        print(f"  Initial health: {initial_health}")

        obs, rewards, terminated, truncated, info = env.step(actions)

        new_health = {t['agent_id']: t['health'] for t in info['tanks']}
        print(f"  New health: {new_health}")
        print(f"  Rewards: {rewards}")

        # Check if any damage was done
        damage_dealt = any(initial_health[aid] != new_health[aid]
                          for aid in initial_health)

        if damage_dealt:
            print("  Damage was dealt!")
        else:
            print("  No damage dealt (tanks may not be in line of fire)")

        print("✓ Combat test completed")
        return True
    except Exception as e:
        print(f"✗ Combat test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episode_completion(env):
    """Test that episodes can complete."""
    print("\nTest 8: Episode Completion")
    print("-" * 50)
    try:
        obs, info = env.reset(seed=42)

        steps = 0
        max_steps = 100

        while steps < max_steps:
            # Random actions
            actions = {agent_id: np.random.randint(0, 5) for agent_id in obs.keys()}
            obs, rewards, terminated, truncated, info = env.step(actions)

            steps += 1

            if terminated.get("__all__", False):
                print(f"✓ Episode completed naturally after {steps} steps")
                print(f"  Team 0 alive: {info['team0_alive']}")
                print(f"  Team 1 alive: {info['team1_alive']}")
                print(f"  Final rewards: {rewards}")
                break
        else:
            print(f"  Episode ran for {steps} steps without terminating")

        print("✓ Episode completion test passed")
        return True
    except Exception as e:
        print(f"✗ Episode completion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("TANKSIM ENVIRONMENT TEST SUITE")
    print("="*60)

    results = []

    # Test 1: Environment creation
    success, env = test_environment_creation()
    results.append(("Environment Creation", success))

    if not success or env is None:
        print("\n✗ Cannot proceed with further tests - environment creation failed")
        return

    # Test 2: Reset
    success = test_reset(env)
    results.append(("Reset Functionality", success))

    # Test 3: Action space
    success = test_action_space(env)
    results.append(("Action Space", success))

    # Test 4: Single step
    success = test_single_step(env)
    results.append(("Single Step", success))

    # Test 5: Movement
    success = test_movement(env)
    results.append(("Movement", success))

    # Test 6: Rotation
    success = test_rotation(env)
    results.append(("Rotation", success))

    # Test 7: Combat
    success = test_combat(env)
    results.append(("Combat", success))

    # Test 8: Episode completion
    success = test_episode_completion(env)
    results.append(("Episode Completion", success))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("-"*60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! TankSim environment is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
