from xarm.wrapper import XArmAPI
import time

arm=XArmAPI("192.168.1.222")

print("connected:",arm.connected)
print("pos:",arm.get_position())
print("err:",arm.get_err_warn_code())

arm.clean_warn()
arm.clean_error()

arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)
time.sleep(1)

print("state:",arm.get_state())
print("pos:",arm.get_position())
print("err:",arm.get_err_warn_code())

input("Press Enter to move up 5 mm...:")

arm.set_position(z=5,relative=True,speed=10,wait=True)

print("new pos:",arm.get_position())

arm.disconnect()
