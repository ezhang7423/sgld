from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import numpy as np

from diffuser.models.temporal import TemporalUnet
from diffuser.utils.arrays import report_parameters
from diffuser.datasets.sequence import SequenceDataset
import torch.utils.data as tdata
import pytorch_lightning as pl
import gym
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageOps, ImageChops

# from diffuser.datasets import sequence_dataset

import pickle

dataset = pickle.load(open("mc.pkl", "rb"))


# class MountainCarDataModule(pl.LightningDataModule):
#     def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
#         super().__init__(train_transforms, val_transforms, test_transforms, dims)

#     def prepare_data(self) -> None:

#         return super().prepare_data()
env = gym.make("MountainCarContinuous-v0")
d = SequenceDataset(ds=dataset, env="MountainCarContinuous-v0")
ld = tdata.DataLoader(d, batch_size=3, num_workers=1, shuffle=True, pin_memory=True)

# # model = TemporalUnet(64, 3, 2, 1, dim_mults=(1, 2))
# # print(model)
# # report_parameters(model)


def cycle(dl):
    while True:
        for data in dl:
            yield data


batch = next(cycle(ld))
traj = batch.trajectories[0]

env.unwrapped.state = traj[0, 1:]

render_env = gym.wrappers.Monitor(env, "test", force=True)
render_env.reset()
render_env.unwrapped.state = traj[0, 1:]
recorded = [traj[0].numpy()]
for _, i in enumerate(traj[1:]):
    print(_)
    act = i[:1].numpy()
    obs, rew, done, info =render_env.step(act)
    if done:
        break
    recorded.append(np.concatenate([act, obs]))

render_env.close()

print(traj[:20])
print('\n\n\n\n')
print(np.array(recorded))