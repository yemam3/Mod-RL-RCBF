import random
import numpy as np

class ReplayMemory:

    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask, t=None, next_t=None, cbf_info=None, next_cbf_info=None):

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask, t, next_t, cbf_info, next_cbf_info)
        self.position = (self.position + 1) % self.capacity

    def batch_push(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, t_batch=None, next_t_batch=None, cbf_info_batch=None, next_cbf_info_batch=None):

        for i in range(state_batch.shape[0]):  # TODO: Optimize This
            t_ = t_batch[i] if t_batch is not None else None
            next_t_ = next_t_batch[i] if next_t_batch is not None else None
            cbf_info_ = cbf_info_batch[i] if cbf_info_batch is not None else None
            next_cbf_info_ = next_cbf_info_batch[i] if next_cbf_info_batch is not None else None
            self.push(state_batch[i], action_batch[i], reward_batch[i], next_state_batch[i], mask_batch[i], t_, next_t_, cbf_info_, next_cbf_info_)  # Append transition to memory

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask, t, next_t, cbf_info, next_cbf_info = map(np.stack, zip(*batch))
        return state, action, reward, next_state, mask, t, next_t, cbf_info, next_cbf_info

    def __len__(self):
        return len(self.buffer)
