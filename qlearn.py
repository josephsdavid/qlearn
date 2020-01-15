import pickle
import numpy as np
import math
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD, Adam

def randx(n, grid_size):
    return np.random.randint(n, grid_size-n + 1, size = 1)

class fruit_basket(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        # fruit starts at zero!
        fruit_y = 0
        # fruit x
        # fruit_x = randx(0, self.grid_size - 1)
        fruit_x = np.random.randint(0,self.grid_size, size = 1)
        # basket center
        basket_center = np.random.randint(0, self.grid_size)
        # basket_center = 0
        self.position_vector = np.asarray([fruit_x, fruit_y, basket_center], dtype = np.int16)

    def _update(self, action):
        # basket moves
        self.position_vector[-1] = min(max(1, self.position_vector[-1] + action), self.grid_size -2)
        # fruit falls
        self.position_vector[1] += 1

class game_board(fruit_basket):
    def __init__(self, grid_size):
        super().__init__(grid_size)
        # here we draw the game board
        self.im_size = (self.grid_size,)*2
        canvas = np.zeros(self.im_size)
        canvas[self.position_vector[1] +1, self.position_vector[0]] = 1
        canvas[-1, self.position_vector[-1]-1:self.position_vector[-1] + 2] = 1
        # gpu friendly small int
        self.board = canvas

    def _update(self, action):
        super()._update(action)
        canvas = np.zeros(self.im_size)
        # redraw the board
        canvas[self.position_vector[1], self.position_vector[0]] = 1
        canvas[-1, self.position_vector[-1]-1:self.position_vector[-1] + 2] = 1
        self.board = canvas

    def flatten(self):
        # return flat version for a happy tensorflow
        return self.board.reshape((1,-1))

class game(game_board):
    def __init__(self, grid_size, actions =  np.arange(3) - 1):
        super().__init__(grid_size)
        # actions by default are left stay right
        self.actions = actions
        self.game_over = False
        self.caught_fruit = False

    def at_bottom(self):
        return self.position_vector[1] == self.grid_size - 1

    def in_basket(self):
        pun = self.position_vector[0] - self.position_vector[-1]
        if pun == 0 or pun == 1 or pun == -1:
            truth = True
        else:
            truth = False
        return truth, abs(pun)

    def _update(self, action):
        super()._update(action)
        self.game_over = self.at_bottom()
        self.caught_fruit, punishment = self.in_basket()
        print(punishment)
        # determine if we got it or not!
        if self.game_over:
            # ternary operators too cool
            self.reward = 1 if (self.caught_fruit) else -punishment
        else:
            self.reward = 0
            # try this out
            # punish it for being  off, this should work
            # also see if being we reverse it to 1-distance/grid_size
            # I think this will have issues with my code overall, and ``

#            self.reward = - (abs(self.position_vector[0] - self.position_vector[-1]))/self.grid_size


    def step(self, action):
        # initial state is the flat one, this is an input to the network
        init_state = self.flatten()
        # update
        self._update(action)
        # return some stuff
        return [init_state, action, self.reward, self.flatten(), self.game_over]

class player(game):
    def __init__(self, grid_size, hidden_dim, learning_rate,
                 batch_size, max_memory, discount,
                 epsilon = 0.1, decay_rate = 512, actions = np.arange(3) - 1):

        super().__init__(grid_size, actions)
        inputs = Input((self.grid_size**2,))
        hidden = Dense(hidden_dim[0], activation = 'relu')(inputs)
        for d in hidden_dim[1:]:
            hidden = Dense(d, activation = 'relu')(hidden)
        outputs = Dense(self.actions.shape[0])(hidden)
        self.model = Model(inputs, outputs)
        self.model.compile(SGD(lr=learning_rate), "mse")
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.memory = np.zeros((self.max_memory, 5)).astype(object)
        self.discount = discount
        self.mem_count = 0
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.win_count = 0
        self.epoch = 1
        self.derivative = []
        self.board_history = []

    def step(self, action):
        # do a step of game
        result = super().step(action)
        # flexing on them
        self.memory[self.mem_count % self.max_memory, :] = result
        self.mem_count += 1

    def make_batch(self):
        mem_size = min(self.mem_count, self.max_memory)
        env_dim = self.grid_size **2
        inputs = np.zeros((min(mem_size, self.batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], self.actions.shape[0]))
        # sample memory and pop in to make a batch
        # I want to write this vectorized so I can have dynamic discounts
        # or some way to lambda the discount (I want to see if more heavily
        # waiting the far future or something/ generally dont like big ugly
        # loops)
        for i, idx in enumerate(np.random.randint(0, mem_size, size = inputs.shape[0])):
            # this is a stupid variable name
            specific_memory = self.memory[idx]
            inputs[i:i+1] = specific_memory[0]
            previous_result = self.model.predict(specific_memory[0])[0]
            new_result = self.model.predict(specific_memory[3])[0]
            Q_sa = np.max(new_result)
            targets[i] = previous_result
            game_over = specific_memory[-1]
            if game_over:
                targets[i, specific_memory[1] + 1] = specific_memory[2]
            else:
                if callable(self.discount) == True:
                    targets[i, specific_memory[1] + 1] = specific_memory[2] + self.discount(Q_sa)
                else:
                    targets[i, specific_memory[1] + 1] = specific_memory[2] + self.discount * Q_sa
        return [inputs, targets]



    def play_game(self):
        loss = 0.
        while not self.game_over:
            if np.random.rand() <= self.epsilon:
                action = np.random.randint(0, self.actions.shape[0], size = 1) - 1
            else:
                q = self.model.predict(self.flatten())
                action = np.argmax(q[0]) -1 # something something trying to figure out why I need this one

            # some candy
            if (action == -1):
                print("action taken:\033[92m left\033[0m")
            elif (action == 0):
                print("action taken:\033[93m stay\033[0m")
            elif (action == 1):
                print("action taken: \033[91m right\033[0m")
            else:
                print("idiot")

            self.step(action) # so nice
            model_args = self.make_batch()
            loss += self.model.train_on_batch(model_args[0], model_args[1])

            # more candy
            print('\x1bc')
            print("\033[31m Epoch:\033[0m {:03d} | \033[31m Loss:\033[0m {:4f} | \033[31m Wins:\033[0m {}".format(self.epoch, loss, self.win_count))
            pboard = self.board.copy()
            pboard = pboard.astype(str)
            pboard[np.where(pboard == '1.0')] =  '*'
            pboard[np.where(pboard == '0.0')] =  ' '
            print(pboard)
            print(self.epsilon)
            self.board_history += [self.board]

    def learn_game(self, epochs):
        ep0 = self.epsilon
        for e in range(epochs):
            super().__init__(self.grid_size, self.actions)
            self.play_game()
            if self.caught_fruit:
                self.win_count += 1
            self.epoch += 1
            self.derivative.append(self.win_count / self.epoch)
            self.model.save("model.h5", overwrite=True)
            # exponential decay
            # this is the time for it to half over 100 epochs, when trained at
            # 1000 epochs
            # I dont know if this makes a better model or not but I like the
            # idea of it
            self.epsilon = ep0 * math.exp(-self.epoch/(epochs/math.log(self.decay_rate)))

def player_save(p: player, model_file: str="model.h5", obj_file: str="player.pkl"):
    mod = p.model
    p.model = None
    filehandler = open(obj_file,'wb')
    pickle.dump(p, filehandler)
    mod.save(model_file)
    p.model = mod

def player_load(model_file: str="model.h5", obj_file: str="player.pkl"):
    with open(obj_file, 'rb') as f:
        data = f.read()
        p = pickle.loads(data)
    p.model = load_model(model_file)
    return p

