from copy import deepcopy
import os
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from datetime import datetime
from spinup.algos.pytorch.sad.core import DemoBuffer, ReplayBuffer
import spinup.algos.pytorch.sad.core as core

from spinup.utils.logx import EpochLogger
from tensorboardX import SummaryWriter
from ipdb import set_trace as tt

tf_logger = "logs/sad_"+datetime.now().strftime('%Y%m%d%H%M%S')+'/'
writer = SummaryWriter(logdir=tf_logger)
# python -m spinup.algos.pytorch.sad.sad

def Variable(var):
    return var.to(device)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



test_reward_buffer = []
train_reward_buffer = []
def sad(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=100, demo_file=''):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac).to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    d_params = itertools.chain(ac.d.parameters())
    # Expert replay buffer
    demo_buffer = DemoBuffer()
    demo_buffer.load(demo_file)
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o, a, r, o2, d = Variable(o), Variable(a), Variable(r), Variable(o2), Variable(d)
        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)
            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    dis_criterion = torch.nn.BCELoss()
    def compute_loss_d(sample, demo):
        o_sample, a_sample = Variable(sample['obs']), Variable(sample['act'])
        q1, q2 = ac.q1(o_sample, a_sample), ac.q2(o_sample, a_sample)
        q_fake = torch.min(q1, q2)
        o_demo, a_demo = Variable(demo['obs']), Variable(demo['act'])
        q1d, q2d = ac.q1(o_demo, a_demo), ac.q2(o_demo, a_demo)
        q_real = torch.min(q1d, q2d)
        loss_d = dis_criterion(ac.d(o_sample, a_sample, q_fake),torch.ones(size=(o_sample.shape[0], 1), device=device)) + \
                 dis_criterion(ac.d(o_demo, a_demo, q_real), torch.zeros(o_demo.shape[0], 1, device=device))
        d_info = dict(DisLoss=loss_d.item())
        return loss_d , d_info


    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        o = Variable(o)
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        # q_pi = torch.min(q1_pi, q2_pi)
        d1, d2 = ac.d(o, pi, q1_pi), ac.d(o, pi, q2_pi)
        # TODO: I choose the q that has a smaller discriminator output, which is more similar to the demo Q
        cat_q = torch.cat([q1_pi.unsqueeze(1), q2_pi.unsqueeze(1)], dim=1)
        cat_d = torch.cat([d1, d2], dim=1)
        idx = torch.argmin(cat_d, dim=1).unsqueeze(1)
        q_pi = cat_q.gather(1, idx).squeeze()
        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    d_optimizer = Adam(ac.d.parameters(), lr=lr)
    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False
        for p in d_params:
            p.requires_grad = False
        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True
        for p in d_params:
            p.requires_grad = True
        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def update_discriminator(data, demo):
        # update Discriminator
        d_optimizer.zero_grad()
        for p in q_params:
            p.requires_grad = False

        loss_d, d_info = compute_loss_d(data, demo)
        loss_d.backward()

        # update Q function
        for p in q_params:
            p.requires_grad = True
        for p in d_params:
            p.requires_grad = False
        d_optimizer.zero_grad()
        loss_q_d, _ = compute_loss_d(data, demo)
        loss_q_target = - loss_q_d
        loss_q_target.backward()
        for p in d_params:
            p.requires_grad = True

    # all_obs, all_action = torch.tensor(demo_buffer.obs_buf, device=device), demo_buffer.act_buf
    # stack = []
    # global counter
    # counter = 0
    def get_action(o, deterministic=False):
        o = torch.as_tensor(o, device=device, dtype=torch.float32)
        # norm = torch.norm(all_obs - o, dim=1)
        # idx_min = torch.argmin(norm)
        # if norm[idx_min] < 0.3:
        #     action = all_action[idx_min.item()]
        #     stack.append(1)
        # else:
        action = ac.act(o, deterministic).cpu().numpy()
        return action

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            writer.add_scalar(tag='test_reward', scalar_value=ep_ret, global_step=t)
            test_reward_buffer.append((t, ep_ret))
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            writer.add_scalar(tag='train_reward', scalar_value=ep_ret, global_step=t)
            train_reward_buffer.append((t, ep_ret))
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch1 = replay_buffer.sample_batch(batch_size)
                batch2 = demo_buffer.sample_batch(batch_size)
                #  batch = core.merge_batch(batch1, batch2)
                update(data=batch1)
                update_discriminator(batch1, batch2)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)
                output_dir = logger_kwargs['output_dir']+'/'
                test_rewards = np.array(test_reward_buffer)
                train_rewards = np.array(train_reward_buffer)
                train_file_name = os.path.join(output_dir,'{}_train_rewards.npy'.format(seed))
                test_file_name = os.path.join(output_dir,'{}_test_rewards.npy'.format(seed))
                np.save(train_file_name, train_rewards)
                np.save(test_file_name, test_rewards)
            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('Demo action', len(stack))
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sad-test')
    parser.add_argument('--demo-file', type=str, default='data/Ant50e.pickle')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())
    sad(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, demo_file=args.demo_file)
