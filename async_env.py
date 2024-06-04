from multiprocessing import Process, Pipe
import numpy as np

def worker(remote, parent_remote, env):
    parent_remote.close()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ns, r, done, trunc, info = env.step(data)
            if done:
                env.reset()
            remote.send((ns, r, done, trunc, info))
        elif cmd == 'reset':
            s, _ = env.reset()
            remote.send(s)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'render':
            env.render()
        else:
            raise NotImplementedError

class Envs:
    '''
    A vector of multiple environments for asynchronous training
    '''
    def __init__(self, env, name, capacity):
        self.nenvs = capacity
        self.env = env
        self.name = name
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(capacity)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        print('Start processes..')
        for process in self.ps:
            process.daemon = True
            process.start()
            print(f'Process {process} initialized.')
        for remote in self.work_remotes:
            remote.close()

    def action_space(self):
        return self.env.action_space
    
    def observation_space(self):
        return self.env.observation_space

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting=True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        ns, r, done, trunc, info = zip(*results)
        return np.stack(ns), np.stack(r), np.stack(done), np.stack(trunc), np.stack(info)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])
    
    def reset_idx(self, idx):
        self.remotes[idx].send(('reset', None))
        return self.remotes[idx].recv()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.ps:
            process.join()
        self.closed = True

    def render(self):
        self.remotes[0].send(('render', None))