import argparse
import os
import shared_constants
import torch
from datetime import datetime
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import Figure
from stable_baselines3 import PPO

from RacingDroneAviary import RacingDroneAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

register(
  id='RacingDroneAviary-v0',
  entry_point='RacingDroneAviary:RacingDroneAviary'
)

EPISODE_REWARD_THRESHOLD = 1000
DEFAULT_ENV = 'RacingDroneAviary-v0'
DEFAULT_ALGO = 'ppo'
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_CPU = 1
DEFAULT_STEPS = 1000000
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_LOAD = False
DEFAULT_LOAD_PATH = None

class EvalCallbackMod(EvalCallback):
    def __init__(self, 
                 eval_env, 
                 callback_on_new_best, 
                 verbose, 
                 best_model_save_path,
                 log_path,
                 eval_freq=int(2000/DEFAULT_CPU),
                 deterministic=True,
                 render=False):

        self.drone_path = np.empty(shape=(0,3), dtype=np.float64)
        
        super().__init__(eval_env,
                        callback_on_new_best=callback_on_new_best,
                        verbose=verbose,
                        best_model_save_path=best_model_save_path,
                        log_path=log_path,
                        eval_freq=eval_freq,
                        deterministic=deterministic,
                        render=render)
        
    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Plot values (here a random variable)
            drone_path = self.eval_env.envs[0].env.saved_drone_path
            figure, ax = plt.subplots()
            ax.plot(drone_path[:,0], drone_path[:,1])
            ax.set_aspect('equal')
            # Close the figure after logging it
            self.logger.record("drone_trajectory", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        super()._on_step()


def run(
    env=DEFAULT_ENV,
    algo=DEFAULT_ALGO,
    obs=DEFAULT_OBS,
    act=DEFAULT_ACT,
    cpu=DEFAULT_CPU,
    steps=DEFAULT_STEPS,
    load=DEFAULT_LOAD,
    load_exp=DEFAULT_LOAD_PATH,
    output_folder=DEFAULT_OUTPUT_FOLDER
):

    #### Load directory for model ########################################
    if load:
        filename = load_exp
        print("load_exp:", load_exp)
        if os.path.isfile(load_exp+'/success_model.zip'):
            load_path = load_exp+'/success_model.zip'
        elif os.path.isfile(load_exp+'/best_model.zip'):
            load_path = load_exp+'/best_model.zip'
        else:
            print("[ERROR]: no model under the specified path", load_exp)
    else:
        #### Save directory ########################################
        filename = os.path.join(output_folder, 'save-'+env+'-'+algo+'-'+obs.value+'-'+act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(filename):
            os.makedirs(filename+'/')

    # obstaclesCenterPosition = [[0,1.5,1], [0,3,1], [0,4.5,1], [0,6,1], [0,7.5,1]]
    # obstaclesOrientation = [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]
    # obstaclesStd = 0.25
    # gate_width = 1.25
    # initial_xyzs = np.array([[0.0,0.0,1.0]])
    # initial_xyzs_std = 0.2

    # Alpha Pilot track
    obstaclesCenterPosition = [[20.0, 5.0, 3.0], # Obs-1
                               [15.0, 40.0, 3.0], # Obs-2
                               [ 5.0, 30.0, 3.0], # Obs-3
                               [ 0.0, 5.0, 3.0], # Obs-4
                               [-5.0,-10.0, 3.0], # Obs-5
                               [-5.0,-25.0, 3.0], # Obs-6
                               [ 0.0,-30.0, 3.0], # Obs-7
                               [ 5.0,-25.0, 3.0], # Obs-8
                               [ 5.0,-10.0, 3.0], # Obs-9
                               [-10.0, 5.0, 3.0], # Obs-10
                               [-10.0, 35.0, 3.0], # Obs-11
                               ]

    obstaclesOrientation = [[0.0, 0.0, 0.0], # Obs-1
                            [0.0, 0.0, np.pi/4], # Obs-2
                            [0.0, 0.0, np.pi], # Obs-3
                            [0.0, 0.0, np.pi], # Obs-4
                            [0.0, 0.0, np.pi], # Obs-5
                            [0.0, 0.0, 5/4*np.pi], # Obs-6
                            [0.0, 0.0, 3/2*np.pi], # Obs-7
                            [0.0, 0.0, 7/4*np.pi], # Obs-8
                            [0.0, 0.0, 0.0], # Obs-9
                            [0.0, 0.0, 0.0], # Obs-10
                            [0.0, 0.0, 0.0], # Obs-11
                            ]
    obstaclesStd = 0.25
    obstacleOrientationStd = 0.1
    gate_width = 1.25
    initial_xyzs = np.array([[20.0, -25.0, 3.0]])
    initial_xyzs_std = 0.2

    
    sa_env_kwargs = dict(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, 
                         obs=obs, 
                         act=act, 
                         initial_xyzs=initial_xyzs,
                         initial_xyzs_std=initial_xyzs_std,
                         obstaclesCenterPosition=obstaclesCenterPosition,
                         obstaclesOrientation=obstaclesOrientation,
                         obstaclesStd=obstaclesStd,
                         gate_width=gate_width,
                         freq=120,
                         gui=False,
                         )
    train_env = make_vec_env(RacingDroneAviary,
                            env_kwargs=sa_env_kwargs,
                            n_envs=cpu,
                            seed=0
                            )
    eval_env = gym.make('RacingDroneAviary-v0',
                            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                            obs=obs,
                            act=act,
                            initial_xyzs=initial_xyzs,
                            initial_xyzs_std=initial_xyzs_std,
                            obstaclesCenterPosition=obstaclesCenterPosition,
                            obstaclesOrientation=obstaclesOrientation,
                            obstaclesStd=obstaclesStd,
                            gate_width=gate_width,
                            freq=240,
                            gui=True)
    # eval_env_kwargs = dict(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, 
    #                      obs=obs, 
    #                      act=act, 
    #                      obstaclesCenterPosition=obstaclesCenterPosition,
    #                      obstaclesOrientation=obstaclesOrientation,
    #                      obstaclesStd=obstaclesStd,
    #                      gate_width=gate_width,
    #                      gui=True,
    #                      )

    # eval_env = make_vec_env(RacingDroneAviary,
    #                         env_kwargs=eval_env_kwargs,
    #                         n_envs=cpu,
    #                         seed=0
    #                         )

    # onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
    #                        net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])])
    onpolicy_kwargs = dict(activation_fn=torch.nn.Tanh,
                           net_arch=[dict(vf=[128, 128], pi=[128, 128])])

    #### Train the model #######################################
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD,
                                                     verbose=1
                                                     )
    eval_callback = EvalCallbackMod(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(2000/cpu),
                                 deterministic=True,
                                 render=False
                                 )

    model = PPO(a2cppoMlpPolicy,
                train_env,
                policy_kwargs=onpolicy_kwargs,
                tensorboard_log=filename+'/tb/',
                verbose=1
                )

    if load:
        print("[INFO] Model {} loaded successfully:".format(filename))
        model = PPO.load(load_path, env=train_env)

    model.learn(total_timesteps=steps, #int(1e12),
                callback=eval_callback,
                log_interval=100,
                reset_num_timesteps=False,
                )

    #### Save the model ########################################
    model.save(filename+'/success_model.zip')


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning experiments script')
    parser.add_argument('--cpu',        default=DEFAULT_CPU,    type=int, help='Number of training environments (default: 1)', metavar='')        
    parser.add_argument('--steps',      default=DEFAULT_STEPS,  type=int, help='Number of training time steps (default: 35000)', metavar='')
    parser.add_argument('--load',       default=DEFAULT_LOAD, action='store_true', help='Load from results')
    parser.add_argument('--load_exp',   default=DEFAULT_LOAD_PATH, type=str, help='Load path', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))