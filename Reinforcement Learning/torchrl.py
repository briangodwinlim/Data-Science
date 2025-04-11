import torch
from torch import nn
import matplotlib.pyplot as plt
from tensordict.nn import TensorDictModule as TensorDict
from tensordict.nn import TensorDictSequential as Sequential
from torchrl.envs import GymEnv, StepCounter, ObservationNorm, Compose, TransformedEnv
from torchrl.modules import QValueModule, EGreedyModule, MLP
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.record import CSVLogger, VideoRecorder
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector

# Define device
device = torch.device('cuda:0')

# Define environment
env = TransformedEnv(
    env=GymEnv('CartPole-v1', device=device),
    transform=StepCounter(),
    # transform=Compose(
    #     ObservationNorm(in_keys=["observation"]), 
    #     StepCounter(),
    # )
)
# env.transform[0].init_stats(num_iter=1000)

# Set seed
env.set_seed(42)
torch.manual_seed(42)

# Define value network and policy (agent)
value_network = TensorDict(
        module=MLP(
        in_features=env.observation_spec['observation'].shape[-1], 
        out_features=env.action_spec.shape[-1], 
        num_cells=[64, 64], 
        activation_class=nn.Tanh,
        device=device,
    ),
    in_keys=['observation'],
    out_keys=['action_value'],
)
policy = Sequential(value_network, QValueModule(spec=env.action_spec))
policy_explore = Sequential(policy, EGreedyModule(spec=env.action_spec, eps_init=0.5, eps_end=0.1, annealing_num_steps=100_000))

# Define collector and replay buffer
collector = SyncDataCollector(
    create_env_fn=env,
    policy=policy_explore,
    frames_per_batch=100,
    total_frames=-1,
    device=device,
    init_random_frames=5000,
)
replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=100_000, device=device))

# Define loss and optimizer
loss = DQNLoss(value_network=policy, delay_value=True, action_space=env.action_spec)
optimizer = torch.optim.Adam(params=loss.parameters(), lr=0.05)
updater = SoftUpdate(loss_module=loss, eps=0.95)

# Training
num_count = 0
num_episodes = 0
success_steps = []
for data in collector:
    replay_buffer.extend(data)
    if len(replay_buffer) > collector.init_random_frames:
        for _ in range(10):
            sample = replay_buffer.sample(batch_size=128)
            loss_dict = loss(sample)
            loss_dict['loss'].backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update exploration factor and target params
            policy_explore[1].step(frames=data.numel())
            updater.step()
            
            # Update logs
            num_count = num_count + data.numel()
            num_episodes = num_episodes + data['next', 'done'].sum()
    
    success_steps.append(replay_buffer[:]['next', 'step_count'].max())

    if num_count > 0 and num_count % 1000 == 0:
        print(f'Successful steps: {success_steps[-1]}, replay buffer length {len(replay_buffer)}, number of episodes: {num_episodes}')

    if success_steps[-1] > 475:
        break

# Plot training
plt.plot(success_steps)
plt.title('Successful steps over training episodes')
plt.xlabel('Training episodes')
plt.ylabel('Steps')
plt.show()

# Save run
recorder = VideoRecorder(logger=CSVLogger(exp_name='DQN', log_dir='./output', video_format='mp4'), tag='video')
record_env = TransformedEnv(env=GymEnv('CartPole-v1', from_pixels=True, pixels_only=False), transform=recorder)
record_env.set_seed(42)
record_env.rollout(max_steps=1000, policy=policy)
recorder.dump()
