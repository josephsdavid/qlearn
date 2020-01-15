import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.python.framework.ops import disable_eager_execution
import os
import qlearn as q
matplotlib.use('agg')
disable_eager_execution()

par_names = ["grid_size", "hidden_dim", "learning_rate",
             "batch_size", "max_memory", "discount",
             "epsilon", "decay_rate", "actions"]

par_vals = [10, (100,100), 0.2, 50, 500, 0.9, 0.9, 512, np.arange(3) - 1]

dave = q.player(**dict(zip(par_names, par_vals)))

dave.learn_game(1000)
q.player_save(dave)
dave.model.summary()

dave = q.player_load()
dave.model.summary()

for i in range(1000):
    print(i)
    mat = dave.board_history[:8000][i]
    plt.imshow(mat.reshape((dave.grid_size,)*2),
               interpolation = 'none', cmap = 'gray')
    plt.savefig("{:03d}.png".format(i))
    plt.clf()
    plt.close()
os.system("ffmpeg -i %03d.png result.gif -vf fps=1 -y "   )
os.system("rm *.png")

plt.plot(dave.derivative)
plt.savefig("training_history.png")
