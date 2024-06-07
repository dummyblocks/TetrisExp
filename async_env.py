from multiprocessing import Process, Pipe
import numpy as np
from copy import deepcopy

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
        else:
            raise NotImplementedError

class SubprocEnvs:
    '''
    A vector of multiple environments for asynchronous training
    '''
    def __init__(self, envs, name):
        self.nenvs = len(envs)
        self.env = envs[0]
        self.name = name
        self.waiting = False
        self.closed = False
        self.recent = None
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env))
                   for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]
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
        recent = list(ns)
        for i, _done in enumerate(done):
            if _done:
                recent[i] = self.reset_idx(i)
        self.recent = np.stack(recent)
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
    
    def get_recent(self):
        if self.recent is not None:
            return self.recent
        return self.reset()

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

class SerialEnvs:
    '''
    A vector of multiple environments for multi agent training
    '''
    def __init__(self, envs, name):
        self.nenvs = len(envs)
        self.env = envs[0]
        self.recent = None
        self.venv = envs
        self.name = name
        self.closed = False
        print('Loaded envs.')

    def action_space(self):
        return self.env.action_space
    
    def observation_space(self):
        return self.env.observation_space

    def step(self, actions):
        ns, r, done, trunc, info, recent = [], [], [], [], [], []
        i = 0
        for i, a in enumerate(actions):
            _ns, _r, _done, _trunc, _info = self.venv[i].step(a)
            ns.append(_ns)
            r.append(_r)
            done.append(_done)
            trunc.append(_trunc)
            info.append(_info)
            recent.append(_ns if not _done else self.reset_idx(i))
        
        self.recent = np.stack(recent)
        return np.stack(ns), np.stack(r), np.stack(done), np.stack(trunc), np.stack(info)
        
    def reset(self):
        inits = []
        for i in range(self.nenvs):
            inits.append(self.venv[i].reset()[0])
        self.recent = np.stack(inits)
        return self.recent
    
    def reset_idx(self, idx):
        return self.venv[idx].reset()[0]
    
    def get_recent(self):
        if self.recent is not None:
            return self.recent
        return self.reset()

    def close(self):
        for i in range(self.nenvs):
            self.venv[i].close()
        self.closed = True