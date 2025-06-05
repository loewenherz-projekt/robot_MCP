"""
Script to continuously print the current positions of all motors and and human readable state.
Updates 10 times per second without sending any commands to the robot.
"""

import time
import argparse
from robot_controller import RobotController

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Display robot positions at regular intervals.')
    parser.add_argument('--rate', type=float, default=10.0, 
                        help='Update rate in Hz (default: 10.0)')
    args = parser.parse_args()
    
    # Calculate sleep time from rate
    sleep_time = 1.0 / args.rate
    
    try:
        # Initialize robot controller with all motors
        controller = RobotController(update_goal_pos=False)
        print(f"Connected to motor bus. Updating at {args.rate} Hz.")
        
        # Main loop
        try:
            while True:
                # Get current robot state
                result = controller.get_current_robot_state()
                if not result.ok:
                    print(f"Error getting robot state: {result.msg}")
                    continue
                
                # Extract data for display
                state = result.robot_state
                
                # Display joint positions
                print("\n--- Joint Positions (degrees) ---")
                for motor, angle in state['joint_positions_deg'].items():
                    print(f"{motor}: {angle:.2f}")
                
                # Display human-readable state
                print("\n--- Human Readable State ---")
                for key, value in state['human_readable_state'].items():
                    print(f"{key}: {value}")
                
                # Sleep to maintain update rate
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'controller' in locals():
            controller.disconnect(reset_pos=False)
            print("Disconnected from motor bus")

if __name__ == "__main__":
    main() 