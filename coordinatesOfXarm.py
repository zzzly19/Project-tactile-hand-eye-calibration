# to Verify the pose detection data captured of the xarm


from xarm.wrapper import XArmAPI
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class XArmPoseReader:
    def __init__(self, ip='192.168.1.222'):
        """
        Initialize connection to xArm
        :param ip: IP address of the xArm (default: '192.168.1.222')
        """
        self.arm = XArmAPI(ip)
        self.connected = False
        
    def connect(self):
        """Connect to the xArm"""
        self.arm.connect()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Position control mode
        self.arm.set_state(0)  # Sport state
        self.connected = True
        print("xArm connected successfully")
        
    def disconnect(self):
        """Disconnect from the xArm"""
        if self.connected:
            self.arm.disconnect()
            self.connected = False
            print("xArm disconnected")
            
    def get_pose(self, verbose=True):
        """
        Get current end effector pose
        Returns:
            dict: {
                'position_mm': [x, y, z] in mm,
                'position_m': [x, y, z] in meters,
                'orientation_rad': [roll, pitch, yaw] in radians,
                'orientation_deg': [roll, pitch, yaw] in degrees,
                'rotation_matrix': 3x3 rotation matrix,
                'quaternion': [x, y, z, w] quaternion
            }
        """
        if not self.connected:
            raise ConnectionError("Arm not connected")
            
        # Get raw pose data (position in mm, orientation in radians)
        pose_data = self.arm.get_position()[1]
        position_mm = pose_data[:3]
        orientation_rad = pose_data[3:6]  # roll, pitch, yaw
        
        # Convert to other representations
        position_m = [p/1000.0 for p in position_mm]
        orientation_deg = [np.degrees(a) for a in orientation_rad]
        
        # Create rotation matrix and quaternion
        rot = R.from_euler('xyz', orientation_rad)
        rotation_matrix = rot.as_matrix()
        quaternion = rot.as_quat()  # [x, y, z, w]
        
        pose = {
            'position_mm': position_mm,
            'position_m': position_m,
            'orientation_rad': orientation_rad,
            'orientation_deg': orientation_deg,
            'rotation_matrix': rotation_matrix,
            'quaternion': quaternion
        }
        
        if verbose:
            self.print_pose(pose)
            
        return pose
    
    def print_pose(self, pose):
        """Pretty print the pose information"""
        print("\n=== Current xArm Pose ===")
        print(f"Position (mm): X={pose['position_mm'][0]:.2f}, Y={pose['position_mm'][1]:.2f}, Z={pose['position_mm'][2]:.2f}")
        print(f"Position (m): X={pose['position_m'][0]:.4f}, Y={pose['position_m'][1]:.4f}, Z={pose['position_m'][2]:.4f}")
        print(f"Orientation (rad): Roll={pose['orientation_rad'][0]:.4f}, Pitch={pose['orientation_rad'][1]:.4f}, Yaw={pose['orientation_rad'][2]:.4f}")
        print(f"Orientation (deg): Roll={pose['orientation_deg'][0]:.2f}, Pitch={pose['orientation_deg'][1]:.2f}, Yaw={pose['orientation_deg'][2]:.2f}")
        print("\nRotation Matrix:")
        print(np.array2string(pose['rotation_matrix'], precision=4, suppress_small=True))
        print(f"\nQuaternion (x,y,z,w): {pose['quaternion']}")
        print("=======================")
        
    def continuous_pose_readout(self, interval=0.5):
        """
        Continuously read and display the arm pose
        :param interval: time between readings in seconds
        """
        try:
            print("Starting continuous pose readout (press Ctrl+C to stop)...")
            while True:
                self.get_pose()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped continuous pose readout")

if __name__ == "__main__":
    # Example usage
    pose_reader = XArmPoseReader()
    
    try:
        # Connect to the arm
        pose_reader.connect()
        
        # Get a single pose reading
        print("\nSingle pose reading:")
        pose = pose_reader.get_pose()
        
        # Continuous readout
        pose_reader.continuous_pose_readout(interval=1.0)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pose_reader.disconnect()