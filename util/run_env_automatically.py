from aux_functions import *
import subprocess, time, os, signal, psutil   # psutil is optional but handy

def restart_UE_env(cfg):

    # Stop enviroment
    stop_environment(cfg.env_process, timeout=10)

    # Start enviroment
    cfg.env_process, cfg.env_folder = start_environment(env_name=cfg.env_name)
    print(f"{cfg.env_process} in {cfg.env_folder} has been started")

    # Reconnect drone
    client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode, num_agents=cfg.num_agents)

    #Restart learning
    automate = True

    return cfg.env_process, cfg.env_folder, automate, client

def stop_environment(env_proc, timeout=10):
    if env_proc.poll() is not None:                  # already dead?
        return
    print("Stopping Unreal Engine …")
    try:
        subprocess.run(["taskkill", "/PID", str(env_proc.pid), "/F", "/T"])
        env_proc.wait(timeout=timeout)               # give it a few seconds
        print("✓ Environment closed gracefully")
    except subprocess.TimeoutExpired:
        print("✗ UE did not exit – killing")
