import argparse
import pickle

import numpy as np
import wandb, warnings
import torch
from config import args_khazad_dum_varibad
from config.mujoco import args_cheetah_vel_rl2, args_cheetah_vel_varibad, args_cheetah_mass_varibad, \
    args_cheetah_body_varibad,  args_ant_goal_rl2, args_ant_goal_varibad, args_ant_mass_varibad, \
    args_humanoid_vel_varibad, args_humanoid_mass_varibad, args_humanoid_body_varibad
from environments.parallel_envs import make_vec_envs
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.logger import configure
# from cuquantum_ppo.custom_ppo import CuQuantumPPO
# from cuquantum_ppo.custom_callback import CuQuantumCallback
# from muzero.muzero import MuZero
import logging

import os
def generate_exp_label(args):
    if not args.exp_label:
        if args.oracle:
            method = 'oracle'  # 'Oracle'
        elif args.cem == 0 and args.tail == 0:
            method = 'varibad'  # 'VariBAD'
        elif args.cem == 1:
            method = 'cembad'  # 'RoML'
        elif args.tail == 1:
            method = 'cvrbad'  # 'CVaR_MRL'
        else:
            raise ValueError(args.cem, args.tail)

        env_name_map = {
            'KhazadDum-v0':'kd',
            'HalfCheetahVel-v0':'hcv',
            'HalfCheetahMass-v0':'hcm',
            'HalfCheetahBody-v0':'hcb',
            'HumanoidVel-v0':'humv',
            'HumanoidMass-v0':'humm',
            'HumanoidBody-v0':'humb',
            'AntMass-v0': 'antm',
        }
        env_name = env_name_map[args.env_name]

        args.exp_label = f'{env_name}_{method}'

        if isinstance(args.seed, (tuple,list)):
            args.seed = args.seed[0]

    try:
        if args.exp_suffix:
            args.exp_label = f'{args.exp_label}_{args.exp_suffix}'
    except:
        warnings.warn(f'Missing attribute args.exp_label')

    return args

# os.environ["WANDB_MODE"] = "dryrun"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='gridworld_varibad')
    args, rest_args = parser.parse_known_args()
    env = args.env_type


    if env == 'khazad_dum_varibad':
        args = args_khazad_dum_varibad.get_args(rest_args)

    # --- MUJOCO ---

    # - Cheetah -
    elif env == 'cheetah_vel_varibad':
        args = args_cheetah_vel_varibad.get_args(rest_args)
    elif env == 'cheetah_vel_rl2':
        args = args_cheetah_vel_rl2.get_args(rest_args)
    elif env == 'cheetah_mass_varibad':
        args = args_cheetah_mass_varibad.get_args(rest_args)
    elif env == 'cheetah_body_varibad':
        args = args_cheetah_body_varibad.get_args(rest_args)
    #
    # - Humanoid -
    elif env == 'humanoid_vel_varibad':
        args = args_humanoid_vel_varibad.get_args(rest_args)
    elif env == 'humanoid_mass_varibad':
        args = args_humanoid_mass_varibad.get_args(rest_args)
    elif env == 'humanoid_body_varibad':
        args = args_humanoid_body_varibad.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")

    args = generate_exp_label(args)
    # config = setup_experiment(project='RLOptimizer', enable_wandb_update=True)
    env = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                        gamma=args.policy_gamma, device='cpu',
                        episodes_per_task=args.max_rollouts_per_task,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                        tasks=None,
                        )

    ngc_run = os.path.isdir('/ws')
    if ngc_run:
        ngc_dir = '/result/wandb/'  # args.ngc_path
        os.makedirs(ngc_dir, exist_ok=True)
        logging.info('NGC run detected. Setting path to workspace: {}'.format(ngc_dir))
        wandb.init(project="roml", sync_tensorboard=True, config=args, dir=ngc_dir)
    else:
        wandb.init(project="roml", sync_tensorboard=True, config=args)
    logger = configure(wandb.run.dir, ["stdout", "tensorboard"])
    # n_steps = config['ppo']['steps_per_epoch_per_env']
    # total_timesteps = config['ppo']['total_steps']
    train_algo = PPO  # if learning_method == 'ppo' else CuQuantumPPO
    # callback = CustomEvalCallback(config, n_eval_episodes=config['eval']['n_eval_episodes'],
    #                               eval_freq=config['eval']['eval_freq'])
    model = PPO(policy=MlpPolicy, env=env, tensorboard_log='./runs', seed=args.seed,
                env_name=args.env_name, cem_alpha=args.alpha if args.cem else 0)
    model.set_logger(logger)
    model.learn(args.num_frames)

    if config['model'].get('pretrained_model', None):
        logging.info(f'Training for {int(total_timesteps / (n_steps * env.num_envs))} epochs')
        model = model.load(config['model']['pretrained_model'], env=env, env_check=False,
                           tensorboard_log='./runs', n_steps=n_steps,
                           batch_size=config['learning']['batch_size'], gamma=config['learning']['gamma'],
                           ent_coef=config['learning']['entropy_weight'], vf_coef=config['learning']['value_weight'],
                           seed=config['train_params']['seed'], device=config['train_params']['device'])
        stat_file = os.path.join(os.path.dirname(config['model']['pretrained_model']),
                                 'stats' + os.path.basename(config['model']['pretrained_model'])[5:])
        stats = pickle.load(open(stat_file, 'rb'))
    if config['train']['epochs'] > 0:
        logging.info('=============Starting Training=======================')
        model.learn(total_timesteps, log_interval=1, callback=callback)
    else:
        logging.info('=============Starting Evaluation=======================')
        total_timesteps, callback = model._setup_learn(total_timesteps=0, eval_env=None, callback=callback,
                                                       eval_freq=config['eval']['eval_freq'], tb_log_name='PPO')
        callback._on_training_start(external_constant=stats['reward_normalization_factor'])
        config['eval']['reset_eval_env_after_evaluation'] = False
        config['train_params']['sync_mode'] = 'eval_envs'
        callback.counter = config['eval']['long_eval_freq']
        if config['baselines']['policy_eval_seed'] < 0:
            logging.info('Loading model random generator state')
            if 'numpy_rng_state' in stats:
                np.random.set_state(stats['numpy_rng_state'])
            if 'torch_rng_state' in stats:
                load_torch_rng_state(stats['torch_rng_state'][0], gpu_rng_state=stats['torch_rng_state'][1],
                                     device=model.device)
        else:
            logging.info(f"Setting random seed to {config['baselines']['policy_eval_seed']}")
            np.random.seed(config['baselines']['policy_eval_seed'])
            torch.random.manual_seed(config['baselines']['policy_eval_seed'])
        callback._evaluate_model()
        callback.update_best_result()
        best_short_eval_result = abs(np.mean([x['total_reward'] for x in model.best_results['eval']]))
        best_long_eval_result = abs(np.mean([x['total_reward'] for x in model.best_results['long_eval']]))
        best_result = min(best_short_eval_result, best_long_eval_result)
        logging.info(f"Total flops: {best_result}")
