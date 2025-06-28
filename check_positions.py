#!/usr/bin/env python3
"""
Script to continuously monitor robot positions.
NEVER sends any commands to the robot - pure read-only monitoring.
Updates continuously while you move the robot manually.
"""

import sys
import time
import logging
from robot_controller import RobotController
from config import robot_config

# Configure logging only if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def clear_screen():
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def print_robot_state(controller):
    """Print current robot state in all formats."""
    result = controller.get_current_robot_state()
    
    if not result.ok:
        print(f"❌ Error getting robot state: {result.msg}")
        return False
    
    state = result.robot_state
    
    # Header
    print("=" * 80)
    print(f"🤖 ROBOT POSITION MONITOR - {controller.robot_type.upper()}")
    print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')} | 🔓 TORQUE DISABLED - Move robot manually!")
    print("=" * 80)
    
    # Joint positions in both formats
    print("\n📊 JOINT POSITIONS")
    print("-" * 60)
    print(f"{'Joint Name':<18} | {'Degrees':<10} | {'Normalized':<12}")
    print("-" * 60)
    
    for joint_name in sorted(controller.joint_names):
        deg_val = state["joint_positions_deg"][joint_name]
        norm_val = state["joint_positions_norm"][joint_name]
        print(f"{joint_name:<18} | {deg_val:>8.1f}°  | {norm_val:>10.1f}")
    
    # Cartesian coordinates
    print(f"\n🎯 CARTESIAN COORDINATES")
    print("-" * 30)
    cartesian = state["cartesian_mm"]
    print(f"X (forward/back): {cartesian['x']:>8.1f} mm")
    print(f"Z (up/down):      {cartesian['z']:>8.1f} mm")
    
    # Human-readable state
    print(f"\n🎮 HUMAN-READABLE STATE")
    print("-" * 50)
    human_state = state["human_readable_state"]
    for key, value in human_state.items():
        unit = "mm" if "mm" in key else ("%" if "pct" in key else "°")
        print(f"{key:<30}: {value:>8.1f} {unit}")
    
    # Motor value ranges (reference)
    print(f"\n📋 MOTOR VALUE RANGES (Reference)")
    print("-" * 60)
    print(f"{'Joint Name':<18} | {'Norm Range':<15} | {'Degree Range'}")
    print("-" * 60)
    
    for joint_name, (norm_min, norm_max, deg_min, deg_max) in robot_config.MOTOR_NORMALIZED_TO_DEGREE_MAPPING.items():
        print(f"{joint_name:<18} | {norm_min:>4.0f} to {norm_max:>4.0f}   | {deg_min:>6.1f}° to {deg_max:>6.1f}°")
    
    print(f"\n💡 Press Ctrl+C to exit")
    return True

def main():
    """Main test function."""
    print("🚀 Starting Robot Position Monitor...")
    print("⚠️  READ-ONLY MODE: Robot will connect and then DISABLE TORQUE")
    print("🔓 After connection, you can move the robot manually")
    print("📖 Move your robot manually to see position updates")
    
    try:
        # Initialize robot controller in READ-ONLY mode
        print("\n🔌 Connecting to robot...")
        with RobotController(read_only=True) as controller:
            print(f"✅ Connected to {controller.robot_type}")
            print("🔓 TORQUE DISABLED: Robot can now be moved manually!")
            print("📊 Starting position monitoring...")
            
            # Continuous monitoring loop
            while True:
                clear_screen()
                
                if not print_robot_state(controller):
                    print("❌ Failed to get robot state. Retrying in 1 second...")
                    time.sleep(1)
                    continue
                
                # Update every 100ms for responsive monitoring
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        clear_screen()
        print("\n👋 Robot position monitoring stopped by user.")
        print("✅ No movement commands were sent to the robot.")
        print("🔓 Torque was disabled - robot was free to move manually.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during position monitoring: {e}")
        print("\n🔍 TROUBLESHOOTING:")
        print("1. Make sure robot is connected to the correct port")
        print("2. Check that no other program is using the robot")
        print("3. Verify robot is powered on")
        print("4. Check USB cable connection")
        logging.error(f"Error during position monitoring: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 