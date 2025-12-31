from aux_functions import *

import subprocess, signal, os, time, psutil

CREATE_NEW_PROCESS_GROUP = 0x00000200  # Windows flag

def start_environment(env_name="indoor_twist"):
    env_folder = os.path.join(os.path.dirname(__file__),
                              "unreal_envs", env_name)
    exe_path   = os.path.join(env_folder, f"{env_name}.exe")

    env_proc = subprocess.Popen(
        exe_path,
        creationflags=CREATE_NEW_PROCESS_GROUP)

    time.sleep(5)
    return env_proc, env_folder

def stop_environment(env_proc, timeout=10):
    try:
        subprocess.run(["taskkill", "/PID", str(env_proc.pid), "/F", "/T"])
    except subprocess.TimeoutExpired:
        print("Cannot kill process")
        env_proc.kill()                  # hard kill fallback


if __name__ == "__main__":
    env_proc, _ = start_environment()
    # Option A: taskkill
    # subprocess.run(["taskkill", "/PID", str(env_proc.pid), "/F", "/T"])
    # print("Is it working?")