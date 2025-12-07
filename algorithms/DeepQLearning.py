# Author: Aqeel Anwar(ICSRL)
# Created: 2/19/2020, 8:39 AM
# Email: aqeel.anwar@gatech.edu

import sys, cv2
import nvidia_smi
from network.agent import PedraAgent
from unreal_envs.initial_positions import *
from os import getpid
from network.Memory import Memory
from aux_functions import *
import os
from util.transformations import euler_from_quaternion
from configs.read_cfg import read_cfg, update_algorithm_cfg

import tensorflow as tf
from tensorflow import keras
from util.object_detector import detect_object, object_is_captured, visualize_object_detection
from util.run_env_automatically import restart_UE_env


def DeepQLearning(cfg, env_process, env_folder):

    TensorBoard = keras.callbacks.TensorBoard
    # log_dir = 'models/trained/Indoor/indoor_cloud/Imagenet/e2e/profile'
    log_dir = 'logs/profile'
    # tensorboard_callback = TensorBoard(log_dir=log_dir, profile_batch='10,15')

    algorithm_cfg = read_cfg(config_filename='configs/DeepQLearning.cfg', verbose=True)
    algorithm_cfg.algorithm = cfg.algorithm

    if 'GlobalLearningGlobalUpdate-SA' in algorithm_cfg.distributed_algo:
        # algorithm_cfg = update_algorithm_cfg(algorithm_cfg, cfg)
        cfg.num_agents = 1

    # # Start the environment
    # env_process, env_folder = start_environment(env_name=cfg.env_name)
    # Connect to Unreal Engine and get the drone handle: client
    client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode, num_agents=cfg.num_agents)
    initial_pos = old_posit.copy()
    # Load YOLO
    detect_object(np.zeros((10,10,3), dtype=np.uint8), use_yolo=True)   # warm-up YOLO
    # Load the initial positions for the environment
    reset_array, reset_array_raw, level_name, crash_threshold = initial_positions(cfg.env_name, initZ, cfg.num_agents)

    # Initialize System Handlers
    process = psutil.Process(getpid())
    # nvidia_smi.nvmlInit()

    # Load PyGame Screen
    screen = pygame_connect(phase=cfg.mode)

    fig_z = []
    fig_nav = []
    debug = False
    # Generate path where the weights will be saved
    cfg, algorithm_cfg = save_network_path(cfg=cfg, algorithm_cfg=algorithm_cfg)
    current_state = {}
    new_state = {}
    posit = {}
    name_agent_list = []
    agent = {}
    # Replay Memory for RL
    if cfg.mode == 'train':
        ReplayMemory = {}
        target_agent = {}

        if algorithm_cfg.distributed_algo == 'GlobalLearningGlobalUpdate-MA':
            print_orderly('global', 40)
            # Multiple agent acts as data collecter and one global learner
            global_agent = PedraAgent(algorithm_cfg, client, name='DQN', vehicle_name='global')
            ReplayMemory = Memory(algorithm_cfg.buffer_len)
            target_agent = PedraAgent(algorithm_cfg, client, name='Target', vehicle_name='global')
            target_agent.network_model.model.set_weights(global_agent.network_model.model.get_weights())
            print('[Init Sync] global → target')

        for drone in range(cfg.num_agents):
            name_agent = "drone" + str(drone)
            name_agent_list.append(name_agent)
            print_orderly(name_agent, 40)
            # TODO: turn the neural network off if global agent is present
            agent[name_agent] = PedraAgent(algorithm_cfg, client, name='DQN', vehicle_name=name_agent)

            if algorithm_cfg.distributed_algo != 'GlobalLearningGlobalUpdate-MA':
                ReplayMemory[name_agent] = Memory(algorithm_cfg.buffer_len)
                target_agent[name_agent] = PedraAgent(algorithm_cfg, client, name='Target', vehicle_name=name_agent)
                target_agent[name_agent].network_model.model.set_weights(agent[name_agent].network_model.model.get_weights())
                print('[Init Sync] online → target (per agent)')
            current_state[name_agent] = agent[name_agent].get_state()

    elif cfg.mode == 'infer':
        # ─────────────────────────────────────────────
        # Evaluation mode: multiple episodes per start position
        # ─────────────────────────────────────────────

        print_orderly("NETWORK PATH (used in infer)", 80)
        print("algorithm_cfg.network_path =", algorithm_cfg.network_path)
        print("[DEBUG] input_size =", algorithm_cfg.input_size)
        print("[DEBUG] num_actions =", algorithm_cfg.num_actions)

        name_agent = 'drone0'
        name_agent_list.append(name_agent)
        agent[name_agent] = PedraAgent(algorithm_cfg, client, name='DQN',vehicle_name=name_agent)

        # Default evaluation hyperparameters.
        # If there are scalar values in cfg, we try to use them; otherwise we fall back to defaults.
        # default_eval_episodes_per_pos = 3
        # default_max_steps_per_episode = 300

        # try:
        #     eval_episodes_per_pos = int(getattr(cfg, 'eval_episodes_per_pos', default_eval_episodes_per_pos))
        # except Exception:
        #     eval_episodes_per_pos = default_eval_episodes_per_pos

        # try:
        #     max_steps_per_episode = int(getattr(cfg, 'max_steps_per_episode', default_max_steps_per_episode))
        # except Exception:
        #     max_steps_per_episode = default_max_steps_per_episode

        # Overwrite cfg fields with plain integers (no DotMap)
        # cfg.eval_episodes_per_pos = eval_episodes_per_pos
        # cfg.max_steps_per_episode = max_steps_per_episode
        cfg.prev_step = None 

        # Use all reset positions defined for this environment
        eval_positions = list(range(len(reset_array[name_agent])))

        print_orderly('EVALUATION MODE (infer) - testing multiple start positions', 80)
        print(f'Positions to test: {eval_positions}')
        print(f'Episodes per position: {algorithm_cfg.eval_episodes_per_pos}, '
            f'max steps per episode: {algorithm_cfg.max_steps_per_episode}\n')

        # Stats structure: per position we track successes, number of episodes and total steps
        eval_stats = {
            pos_idx: {
                'successes': 0,
                'episodes': 0,
                'steps_sum': 0
            }
            for pos_idx in eval_positions
        }

        # Store evaluation state in cfg so we can access it inside the main loop
        cfg.eval_positions = eval_positions
        cfg.eval_stats = eval_stats
        cfg.current_eval_pos_idx = 0
        cfg.current_eval_episode = 0
        cfg.eval_step_in_episode = 0

        # Put the drone on the very first position
        start_pos = cfg.eval_positions[cfg.current_eval_pos_idx]
        reset_to_initial(start_pos, reset_array, client, vehicle_name=name_agent)
        old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)

        # steps counter
        cfg.eval_step_in_episode = 0


        # ───────────── Tensorboard writer initialization for infer mdoe (TensorBoard + text file) ─────────────
        tb_infer_writer = tf.summary.create_file_writer(
            os.path.join(algorithm_cfg.network_path, "inference_logs")
        )

        infer_log_path = os.path.join(algorithm_cfg.network_path, "inference_log.txt")
        infer_logfile = open(infer_log_path, "w")
        infer_logfile.write("pos,episode,step,action,qmax,qmean,state_diff,found,captured,crashed\n")

    # Initialize variables
    iter = 1

    # Totalizers for tensorboard
    iter_totalizer = {}
    captured_totalizer = {}
    seen_totalizer = {}

    # num_collisions = 0
    episode = {}
    active = True

    print_interval = 1
    automate = True
    choose = False
    print_qval = False
    last_crash = {}
    ret = {}
    distance = {}
    num_collisions = {}
    level = {}
    level_state = {}
    level_posit = {}
    times_switch = {}
    last_crash_array = {}
    ret_array = {}
    distance_array = {}
    epi_env_array = {}
    seen_per_level = {}
    captured_per_level = {}
    log_files = {}

    # If the phase is inference force the num_agents to 1
    hyphens = '-' * int((80 - len('Log files')) / 2)
    print(hyphens + ' ' + 'Log files' + ' ' + hyphens)
    for name_agent in name_agent_list:
        ret[name_agent] = 0
        num_collisions[name_agent] = 0
        last_crash[name_agent] = 0
        level[name_agent] = 0
        episode[name_agent] = 0
        iter_totalizer[name_agent] = 1
        captured_totalizer[name_agent] = 0
        seen_totalizer[name_agent] = 0
        level_state[name_agent] = [None] * len(reset_array[name_agent])
        level_posit[name_agent] = [None] * len(reset_array[name_agent])
        times_switch[name_agent] = 0
        last_crash_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        ret_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        distance_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        epi_env_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        seen_per_level[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        captured_per_level[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)        
        distance[name_agent] = 0
        # Log file
        log_path = algorithm_cfg.network_path + '/' + name_agent + '/' + cfg.mode + 'log.txt'
        print("Log path: ", log_path)
        log_files[name_agent] = open(log_path, 'w')

    print_orderly('Simulation begins', 80)

    last_epoch = -1
    restart_request = False

    
    while active:

        try:
            epoch = episode[name_agent]
            if last_epoch != epoch:
                # Call the on_epoch_begin callback to start profiling
                # tensorboard_callback.on_epoch_begin(epoch=epoch)
                last_epoch = epoch

            active, automate, algorithm_cfg, client = check_user_input(active, automate, agent[name_agent], client,
                                                                       old_posit[name_agent], initZ, fig_z, fig_nav,
                                                                       env_folder, cfg, algorithm_cfg)
            


            if automate:

                if iter % algorithm_cfg.ue_restart_interval == 0 or restart_request:
                    cfg.env_process, cfg.env_folder, automate, agent[name_agent].client = restart_UE_env(cfg)
                    client = agent[name_agent].client
                    restart_request = False

                if cfg.mode == 'train':

                    if iter % algorithm_cfg.switch_env_steps == 0:
                        switch_env = True
                    else:
                        switch_env = False

                    for name_agent in name_agent_list:

                        start_time = time.time()
                        if switch_env:
                            posit1_old = client.simGetVehiclePose(vehicle_name=name_agent)
                            times_switch[name_agent] = times_switch[name_agent] + 1
                            level_state[name_agent][level[name_agent]] = current_state[name_agent]
                            level_posit[name_agent][level[name_agent]] = posit1_old
                            last_crash_array[name_agent][level[name_agent]] = last_crash[name_agent]
                            ret_array[name_agent][level[name_agent]] = ret[name_agent]
                            distance_array[name_agent][level[name_agent]] = distance[name_agent]
                            epi_env_array[name_agent][level[name_agent]] = episode[name_agent]

                            level[name_agent] = (level[name_agent] + 1) % len(reset_array[name_agent])

                            print(name_agent + ' :Transferring to level: ', level[name_agent], ' - ',
                                  level_name[name_agent][level[name_agent]])

                            if times_switch[name_agent] < len(reset_array[name_agent]):
                                reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                            else:
                                current_state[name_agent] = level_state[name_agent][level[name_agent]]
                                posit1_old = level_posit[name_agent][level[name_agent]]
                                reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                                client.simSetVehiclePose(posit1_old, ignore_collison=True, vehicle_name=name_agent)
                                time.sleep(0.1)

                            last_crash[name_agent] = last_crash_array[name_agent][level[name_agent]]
                            ret[name_agent] = ret_array[name_agent][level[name_agent]]
                            distance[name_agent] = distance_array[name_agent][level[name_agent]]
                            episode[name_agent] = epi_env_array[name_agent][int(level[name_agent] / 3)]
                            # environ = environ^True
                        else:
                            # TODO: policy from one global agent: DONE
                            if algorithm_cfg.distributed_algo == 'GlobalLearningGlobalUpdate-MA':
                                agent_this_drone = global_agent
                                ReplayMemory_this_drone = ReplayMemory
                                target_agent_this_drone = target_agent
                            else:
                                agent_this_drone = agent[name_agent]
                                ReplayMemory_this_drone = ReplayMemory[name_agent]
                                target_agent_this_drone = target_agent[name_agent]

                            action, action_type, algorithm_cfg.epsilon, qvals = policy(algorithm_cfg.epsilon,
                                                                                       current_state[name_agent], iter,
                                                                                       algorithm_cfg.epsilon_saturation,
                                                                                       algorithm_cfg.epsilon_model,
                                                                                       algorithm_cfg.wait_before_train,
                                                                                       algorithm_cfg.num_actions,
                                                                                       agent_this_drone,
                                                                                       algorithm_cfg.epsilon_override,
                                                                                       algorithm_cfg.epsilon_override_from_iter
                                                                                       )

                            action_word = translate_action(action, algorithm_cfg.num_actions)
                            # Take the action
                            agent[name_agent].take_action(action, algorithm_cfg.num_actions, Mode='static')
                            # time.sleep(0.05)
                            new_state[name_agent] = agent[name_agent].get_state()

                            # Check if the state is valid
                            if new_state[name_agent] is None or new_state[name_agent].size == 0:
                                print("Received invalid state. Skipping this iteration.")
                                continue  # Skip to the next iteration
                            
                            # Detect ball in the new camera frame
                            # 1 Get the RGB frame already stored in new_state[name_agent][0]
                            raw_frame = agent[name_agent].raw_bgr

                            # 2 Run YOLO
                            found, cx, cy, r, bbox = detect_object(raw_frame)

                            # 3 Optionally draw (comment out for speed on headless servers)
                            # visualize_ball_detection(raw_frame, found, cx, cy, r, bbox, name_agent)

                            # 4 Check if object is captured - is big enaugh (drone is close enaugh) and in the center
                            frame_h, frame_w = raw_frame.shape[:2]
                            captured = object_is_captured(found, cx, cy, r,
                                                        frame_w, frame_h,
                                                        centre_frac_x=0.10,
                                                        centre_frac_y=0.30,
                                                        min_radius_frac=0.10)
                            
                            current_level_idx = level[name_agent]
                            # seen_totalizer[name_agent] += 1 if found else 0
                            # captured_totalizer[name_agent] += 1 if captured else 0
                            if found:
                                seen_totalizer[name_agent] += 1
                                # NOWE: licznik dla bieżącej pozycji
                                seen_per_level[name_agent][current_level_idx] += 1

                            if captured:
                                captured_totalizer[name_agent] += 1
                                # NOWE: licznik dla bieżącej pozycji
                                captured_per_level[name_agent][current_level_idx] += 1                            

                            # 5 Compute off-centre error (dx,dy)   (frame is input_size×input_size)
                            if found:
                                dx = cx - frame_w/2
                                dy = cy - frame_h/2
                            else:
                                dx = dy = 0     

                            new_depth1, thresh = agent[name_agent].get_CustomDepth(cfg)

                            # Get GPS information
                            posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                            position = posit[name_agent].position
                            old_p = np.array(
                                [old_posit[name_agent].position.x_val, old_posit[name_agent].position.y_val])
                            new_p = np.array([position.x_val, position.y_val])

                            # calculate distance
                            distance[name_agent] = distance[name_agent] + np.linalg.norm(new_p - old_p)
                            old_posit[name_agent] = posit[name_agent]

                            base_reward, crash = agent[name_agent].reward_gen(new_depth1, action, crash_threshold, thresh, debug, cfg)
                            
                            # ------------------------------------------------------------------ #
                            # Reward shaping
                            # ------------------------------------------------------------------ #
                            bonus = 0.0
                            if found:                                     # we have a YOLO detection
                                # ── (1) centre alignment score  ────────────────────────────────
                                #   1.0  when perfectly centred, 0.0 at far edge
                                centre_score = 1.0 - (abs(dx) + abs(dy)) / frame_w
                                centre_score = max(0.0, centre_score)     # clamp to [0,1]

                                # ── (2) size / distance score  ─────────────────────────────────
                                #   1.0 once the ball’s height ≳ 10 % of the image
                                size_score   = min(1.0, r / (0.10 * frame_h))

                                # ── (3) final bonus  ───────────────────────────────────────────
                                bonus = 10.0 * centre_score * size_score   # 0 … +10

                            # add to baseline AirSim / depth reward
                            total_reward = base_reward + bonus

                            # big jackpot when the capture condition is met
                            if captured:
                                total_reward += 100.0      # ← your “big reward”
                                crash = True              # ← end episode (forces reset)
                            # print(f"[DEBUG] Ball: captured={captured} seen={found}, dx={dx}, dy={dy}, r={r}, reward={total_reward:.3f}")                          
                            # print(f"crash value: {crash}, type: {type(crash)}")

                            agent_state = agent[name_agent].GetAgentState()

                            if agent_state.has_collided or distance[name_agent] < 0.1 or (crash and not captured):
                                num_collisions[name_agent] = num_collisions[name_agent] + 1
                                print('crash')                                
                                total_reward = -1
                                crash = True

                            if last_crash[name_agent] > 150:
                                num_collisions[name_agent] = num_collisions[name_agent] + 1
                                print('Number of steps in episode exceeded')
                                total_reward -= 1
                                crash = True    

                            ret[name_agent] += total_reward                       
                                            
                            data_tuple = []
                            data_tuple.append([current_state[name_agent], action, new_state[name_agent], total_reward, crash])
                            # TODO: one replay memory global, target_agent, agent: DONE
                            err = get_errors(data_tuple,                #1
                                             True,                    #2
                                             ReplayMemory_this_drone,   #3
                                             algorithm_cfg.input_size,  #4
                                             agent_this_drone,          #5
                                             target_agent_this_drone,   #6
                                             algorithm_cfg.gamma,       #7
                                             algorithm_cfg.Q_clip)      #8
                            

                            # ReplayMemory_this_drone.add(err, data_tuple)
                            # ------------------------------------------------------------------
                            # only enqueue when we really have a scalar TD-error
                            # ------------------------------------------------------------------
                            ReplayMemory_this_drone.add(err, data_tuple)

                            # Train if sufficient frames have been stored
                            if iter > algorithm_cfg.wait_before_train:
                                if iter % algorithm_cfg.train_interval == 0:
                                    # Train the RL network
                                    old_states, Qvals, actions, err, idx = minibatch_double(data_tuple,
                                                                                            algorithm_cfg.batch_size,
                                                                                            True,
                                                                                            ReplayMemory_this_drone,
                                                                                            algorithm_cfg.input_size,
                                                                                            agent_this_drone,
                                                                                            target_agent_this_drone,
                                                                                            algorithm_cfg.gamma,
                                                                                            algorithm_cfg.Q_clip)
                                    # TODO global replay memory: DONE
                                    for i in range(algorithm_cfg.batch_size):
                                        ReplayMemory_this_drone.update(idx[i], err[i])
                                    # for i, idxx in enumerate(idx):
                                    #     ReplayMemory.update(idxx, err[i])

                                    if print_qval:
                                        print(Qvals)

                                    t2 = time.time()
                                    # TODO global agent, target_agent: DONE
                                    agent_this_drone.network_model.train_n(old_states, Qvals, actions,
                                                                               algorithm_cfg.batch_size,
                                                                               algorithm_cfg.dropout_rate,
                                                                               algorithm_cfg.learning_rate,
                                                                               algorithm_cfg.epsilon, iter)
                                    
                            time_exec = time.time() - start_time
                            # print(f"ball detection took {t1 - t0:.3f}s, step execution took {time_exec:.3f}s")                                      

                            gpu_memory, gpu_utilization, sys_memory = get_SystemStats(process, cfg.NVIDIA_GPU)

                            for i in range(0, len(gpu_memory)):
                                tag_mem = 'GPU' + str(i) + '-Memory-GB'
                                tag_util = 'GPU' + str(i) + 'Utilization-%'
                                agent[name_agent].network_model.log_to_tensorboard(tag=tag_mem, group='SystemStats',
                                                                                   value=gpu_memory[i],
                                                                                   index=iter)
                                agent[name_agent].network_model.log_to_tensorboard(tag=tag_util, group='SystemStats',
                                                                                   value=gpu_utilization[i],
                                                                                   index=iter)
                            agent[name_agent].network_model.log_to_tensorboard(tag='Memory-GB', group='SystemStats',
                                                                               value=sys_memory,
                                                                               index=iter)

                            s_log = '{:<6s} - Level {:>2d} - Iter: {:>6d}/{:<5d} {:<8s}-{:>5s} Eps: {:<1.4f} ' \
                                    'lr: {:>1.8f} Ret = {:<+6.4f} Last Crash = {:<5d} t={:<1.3f} SF = {:<5.4f}  ' \
                                    'Seen={:<5b} Reward: {:<+1.4f}  '.format(
                                name_agent,
                                int(level[name_agent]),
                                iter,
                                episode[name_agent],
                                action_word,
                                action_type,
                                algorithm_cfg.epsilon,
                                algorithm_cfg.learning_rate,
                                ret[name_agent],
                                last_crash[name_agent],
                                time_exec,
                                distance[name_agent],
                                found,
                                total_reward)

                            if ret[name_agent] == -1 and last_crash[name_agent] == 0 and algorithm_cfg.ue_restart_enabled:
                                restart_request = True

                            if iter % print_interval == 0:
                                print(s_log)
                            log_files[name_agent].write(s_log + '\n')

                            last_crash[name_agent] = last_crash[name_agent] + 1
                            if debug:
                                cv2.imshow(name_agent, np.hstack((np.squeeze(current_state[name_agent], axis=0),
                                                                  np.squeeze(new_state[name_agent], axis=0))))
                                cv2.waitKey(1)

                            if crash:
                                # Call the on_epoch_end callback to stop profiling
                                # tensorboard_callback.on_epoch_end(epoch=epoch)    
                                ################################################## 
                                if distance[name_agent] < 0.01:
                                    # Drone won't move, reconnect
                                    print('Recovering from drone mobility issue')

                                    agent[name_agent].client, old_posit, initZ = connect_drone(
                                        ip_address=cfg.ip_address, phase=cfg.mode,
                                        num_agents=cfg.num_agents, client=client)
                                    time.sleep(2)
                                else:
                                    
                                    # Logs epsiode based
                                    # agent[name_agent].network_model.log_to_tensorboard(tag='Object seen', group=name_agent,
                                    #                                                    value=seen_totalizer[name_agent],
                                    #                                                    index=episode[name_agent])      
                                    # agent[name_agent].network_model.log_to_tensorboard(tag='Object captured', group=name_agent,
                                    #                                                    value=captured_totalizer[name_agent],
                                    #                                                    index=episode[name_agent])  
                                    # pos_idx = level[name_agent]
                                    # tag_seen_pos = f'Object seen pos_{pos_idx}'
                                    # tag_captured_pos = f'Object captured pos_{pos_idx}'
                                    # agent[name_agent].network_model.log_to_tensorboard(tag=tag_seen_pos, group=name_agent,
                                    #                                                     value=seen_per_level[name_agent][pos_idx],
                                    #                                                     index=episode[name_agent])
                                    # agent[name_agent].network_model.log_to_tensorboard(tag=tag_captured_pos, group=name_agent,
                                    #                                                     value=captured_per_level[name_agent][pos_idx],
                                    #                                                     index=episode[name_agent])
                                    # agent[name_agent].network_model.log_to_tensorboard(tag='Return', group=name_agent,
                                    #                                                    value=ret[name_agent],
                                    #                                                    index=episode[name_agent])
                                    # agent[name_agent].network_model.log_to_tensorboard(tag='Safe Flight',
                                    #                                                    group=name_agent,
                                    #                                                    value=distance[name_agent],
                                    #                                                    index=episode[name_agent])
                                    
                                    # Logs step based
                                    logs_step_based = "step_based"
                                    agent[name_agent].network_model.log_to_tensorboard(tag='Object seen', group=logs_step_based,
                                                                                       value=seen_totalizer[name_agent],
                                                                                       index=iter_totalizer[name_agent])      
                                    agent[name_agent].network_model.log_to_tensorboard(tag='Object captured', group=logs_step_based,
                                                                                       value=captured_totalizer[name_agent],
                                                                                       index=iter_totalizer[name_agent])
                                    pos_idx = level[name_agent]
                                    tag_seen_pos = f'Object seen pos_{pos_idx}'
                                    tag_captured_pos = f'Object captured pos_{pos_idx}'

                                    agent[name_agent].network_model.log_to_tensorboard(tag=tag_seen_pos, group=logs_step_based,
                                                                                        value=seen_per_level[name_agent][pos_idx],
                                                                                        index=iter_totalizer[name_agent])
                                    agent[name_agent].network_model.log_to_tensorboard(tag=tag_captured_pos, group=logs_step_based,
                                                                                        value=captured_per_level[name_agent][pos_idx],
                                                                                        index=iter_totalizer[name_agent])                                                                                                   
                                    agent[name_agent].network_model.log_to_tensorboard(tag='Return', group=logs_step_based,
                                                                                       value=ret[name_agent],
                                                                                       index=iter_totalizer[name_agent])
                                    agent[name_agent].network_model.log_to_tensorboard(tag='Safe Flight',
                                                                                       group=logs_step_based,
                                                                                       value=distance[name_agent],
                                                                                       index=iter_totalizer[name_agent])

                                    ret[name_agent] = 0
                                    distance[name_agent] = 0
                                    episode[name_agent] = episode[name_agent] + 1
                                    last_crash[name_agent] = 0

                                    reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                                    # time.sleep(0.2)
                                    current_state[name_agent] = agent[name_agent].get_state()
                                    old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                            else:
                                current_state[name_agent] = new_state[name_agent]

                            if iter % algorithm_cfg.max_iters == 0:
                                automate = False

                    # TODO define and state agents
                    if iter % algorithm_cfg.update_target_interval == 0 and iter > algorithm_cfg.wait_before_train:

                        if algorithm_cfg.distributed_algo == 'GlobalLearningGlobalUpdate-MA':                           
                            target_agent.network_model.model.set_weights(
                                global_agent.network_model.model.get_weights()
                            )
                            print(f'[Target Sync] global → target (iter: {iter})')
                            global_agent.network_model.save_network(algorithm_cfg.network_path, episode[name_agent])
                        else:
                            for name_agent in name_agent_list:
                                agent[name_agent].take_action([-1], algorithm_cfg.num_actions, Mode='static')
                                target_agent[name_agent].network_model.model.set_weights(
                                    agent[name_agent].network_model.model.get_weights()
                                )
                                print(f'[Target Sync] {name_agent}: online → target (iter: {iter})')
                                agent[name_agent].network_model.save_network(algorithm_cfg.network_path,
                                                                             episode[name_agent])                                


                    # if iter % algorithm_cfg.communication_interval == 0 and iter > algorithm_cfg.wait_before_train:
                    #     print('Communicating the weights and averaging them')
                    #     communicate_across_agents(agent, name_agent_list, algorithm_cfg)
                    #     communicate_across_agents(target_agent, name_agent_list, algorithm_cfg)               
                    iter += 1
                    iter_totalizer[name_agent] += 1

                elif cfg.mode == 'infer':
                    # ─────────────────────────────────────────────
                    # Evaluation mode – no learning, only testing
                    # ─────────────────────────────────────────────
                    name_agent = name_agent_list[0]

                    # 1) Check if we finished all positions and episodes
                    if cfg.current_eval_pos_idx >= len(cfg.eval_positions):

                        print_orderly('Evaluation finished', 80)
                        print('\nSummary per start position:')
                        for pos_idx in cfg.eval_positions:
                            stats = cfg.eval_stats[pos_idx]
                            episodes = stats['episodes']
                            if episodes > 0:
                                sr = 100.0 * stats['successes'] / episodes
                                avg_steps = stats['steps_sum'] / episodes
                            else:
                                sr = 0.0
                                avg_steps = 0.0
                            print(f'  Position {pos_idx}: '
                                  f'successes = {stats["successes"]}/{episodes} '
                                  f'({sr:.1f}%), average steps = {avg_steps:.1f}')
                            
                            infer_logfile.write(f'Position {pos_idx}: '
                                  f'successes = {stats["successes"]}/{episodes} '
                                  f'({sr:.1f}%), average steps = {avg_steps:.1f}\n')
                            
                        active = False
                        infer_logfile.write("# Inference finished.\n")
                        infer_logfile.close()
                        tb_infer_writer.close()

                        continue                      


                    # 2) If we are starting a new episode (step 0), reset environment
                    if cfg.eval_step_in_episode == 0:
                        pos_idx = cfg.eval_positions[cfg.current_eval_pos_idx]
                        print(f'\n[Position {pos_idx}] Episode '
                              f'{cfg.current_eval_episode + 1}/{algorithm_cfg.eval_episodes_per_pos}')
                        reset_to_initial(pos_idx, reset_array, client, vehicle_name=name_agent)
                        old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)

                    # 3) Get current state and choose action from the network only (no exploration)
                    current_state[name_agent] = agent[name_agent].get_state()
                    # Check if the state is valid
                    if current_state[name_agent] is None or current_state[name_agent].size == 0:
                        print("[WARN] Invalid state in infer – skipping this step")
                        # do NOT advance eval_step_in_episode in this case
                        continue

                    # --- DEBUG: check how much the state changes over time ---
                    prev = None
                    if hasattr(cfg, "get"):
                        # DotMap / dict – użyj .get, to NIE utworzy pustego DotMapa
                        prev = cfg.get("prev_state", None)
                    else:
                        # awaryjnie – dla zwykłych obiektów
                        prev = getattr(cfg, "prev_state", None)

                    if prev is None or not isinstance(prev, np.ndarray):
                        # first step – just store current state
                        cfg.prev_state = current_state[name_agent].copy()
                        state_diff = 0.0
                    else:
                        state_diff = float(
                            np.mean(np.abs(current_state[name_agent] - prev))
                        )
                        cfg.prev_state = current_state[name_agent].copy()

                    action, action_type, algorithm_cfg.epsilon, qvals = policy(
                        1.0,                             # initial epsilon (ignored in 'inference' model)
                        current_state[name_agent],
                        iter,
                        algorithm_cfg.epsilon_saturation,
                        'inference',                     # epsilon is set to 1.0 inside policy
                        algorithm_cfg.wait_before_train,
                        algorithm_cfg.num_actions,
                        agent[name_agent]
                    )

                    agent[name_agent].take_action(action, algorithm_cfg.num_actions, Mode='static', inference = True)

                    # 4) YOLO: did we see the object and is it "captured"?
                    raw_frame = agent[name_agent].raw_bgr
                    found, cx, cy, r, bbox = detect_object(raw_frame)

                    frame_h, frame_w = raw_frame.shape[:2]
                    captured = object_is_captured(
                        found, cx, cy, r,
                        frame_w, frame_h,
                        centre_frac_x=0.10,
                        centre_frac_y=0.30,
                        min_radius_frac=0.10
                    )

                    a_val = int(action[0]) if hasattr(action, "__len__") else int(action)
                    print(
                        f"[DEBUG] pos={cfg.eval_positions[cfg.current_eval_pos_idx]} "
                        f"ep={cfg.current_eval_episode+1} step={cfg.eval_step_in_episode} | "
                        f"action_type={action_type}, action={a_val}, eps={algorithm_cfg.epsilon:.3f} | "
                        f"state_diff={state_diff:.6f} | "
                        f"Object found={found}"
                    )
           
                    # per-step logging
                    if qvals is not None:
                        qmax = float(np.max(qvals))
                        qmean = float(np.mean(qvals))
                    else:
                        qmax = np.nan
                        qmean = np.nan

                    with tb_infer_writer.as_default():
                        tf.summary.scalar("InferenceStep/Qmax", qmax, step=iter)
                        tf.summary.scalar("InferenceStep/Qmean", qmean, step=iter)
                        
                    # 5) Check for collision
                    agent_state = agent[name_agent].GetAgentState()
                    crashed = agent_state.has_collided

                    cfg.eval_step_in_episode += 1

                    done = False
                    success = False

                    # 6) Episode termination conditions
                    if captured:
                        success = True
                        done = True
                        print("Episode finished successed")
                    elif crashed or cfg.eval_step_in_episode >= algorithm_cfg.max_steps_per_episode:
                        success = False
                        done = True
                        print("crash")

                    # 7) If episode finished – update statistics and move to next one
                    if done:

                        pos_idx = cfg.eval_positions[cfg.current_eval_pos_idx]
                        stats = cfg.eval_stats[pos_idx]
                        stats['episodes'] += 1
                        stats['steps_sum'] += cfg.eval_step_in_episode
                        if success:
                            stats['successes'] += 1

                        print(f'  → END of episode: success={success}, '
                              f'steps={cfg.eval_step_in_episode}')


                        infer_logfile.write(
                                f"pos={cfg.eval_positions[cfg.current_eval_pos_idx]+1}; "
                                f"ep={cfg.current_eval_episode+1}; steps={cfg.eval_step_in_episode}; "
                                f"success={success}\n"
                                )                                          

                        # Prepare for next episode / position
                        cfg.eval_step_in_episode = 0
                        cfg.current_eval_episode += 1

                        # If we exhausted episodes for this position, move to next position
                        if cfg.current_eval_episode >= algorithm_cfg.eval_episodes_per_pos:
                            cfg.current_eval_episode = 0
                            cfg.current_eval_pos_idx += 1

                    # 8) Increase global step counter (for consistency/logging)
                    iter += 1


        except Exception as e:
            if str(e) == 'cannot reshape array of size 1 into shape (0,0,3)':
                print('Recovering from AirSim error')
                client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode,
                                                         num_agents=cfg.num_agents)
                time.sleep(2)
                agent[name_agent].client = client
            else:
                print('------------- Error -------------')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(exc_obj)
                automate = False
                
                print('Hit r and then backspace to start from this point')
