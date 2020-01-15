import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import functools



def initialize(grid_size = 10):
    # shape = (fruit_y, fruit_x, basket_pos)
    fruit_basket = np.asarray([0] + [np.random.randint(i, grid_size-i+1, size = 1) for i in range(2)])
    return fruit_basket.astype(np.int16)

def update(fruit_basket, action, grid_size = 10):
    # center the action on zero
    # update basket position
    basket_pos_new = min(max(1, fruit_basket[-1] + action), grid_size -1)
    # fruit falls one unit
    fruit_basket[0] += 1
    # basket moves in accordance to action
    fruit_basket[-1] = basket_pos_new
    # modify fruit_basket in place!
    return(fruit_basket)


def draw_fruit_basket(fruit_basket, grid_size = 10):
    im_size = (grid_size,)*2
    canvas = np.zeros(im_size)
    # fruit = one pixel, [y,x] = fb[:1]
    canvas[fruit_basket[0], fruit_basket[1] - 1] = 1
    # basket = three pixels wide, center = fb[2]
    canvas[-1, fruit_basket[-1]-1:fruit_basket[-1] + 2] = 1
    #print(canvas.shape)
    return canvas

def at_bottom(y, grid_size=10):
    return y == grid_size-1

def in_basket(x, basket_center):
    return abs(x - basket_center) <=1


def get_reward(fruit_basket, grid_size = 10):
    # (fruity, fruitx, basket_pos)
    bottom = at_bottom(fruit_basket[0], grid_size)
    basket = in_basket(*fruit_basket[1:].T)
    if (bottom):
        if (basket):
            return 1, bottom
        else:
            return -1, bottom
    else:
        return 0, bottom

def get_current_state(fruit_basket, grid_size = 10):
    return draw_fruit_basket(fruit_basket, grid_size).reshape((1,-1))

def one_step(fruit_basket, action, grid_size = 10):
    fruit_basket = update(fruit_basket, action, grid_size)
    reward, over = get_reward(fruit_basket, grid_size)
    return get_current_state(fruit_basket, grid_size), reward, over



# we have defined a single step of basket drawing, updating, and reward
# receiving

# now to do some cool numpy shit

# remember: take in a current memory list, do epic shit

def remember(memory, experience, over, max_memory = 100):
    # lets see if this kills our memory
    # mem = np.zeros(5).reshape(1,5)
    new_experience = np.hstack([experience, over])
    if memory.shape[0] == 1:
        memory = new_experience.reshape(memory.shape)
    elif memory.shape[0] < max_memory:
        memory = np.vstack([memory, new_experience])
    else:
        memory = np.vstack([memory, np.zeros(memory.shape[-1]) ])
        memory[memory.shape[0] % max_memory] = new_experience
    return memory


def get_one_batch(memory, model, batch_size = 10, max_memory = 100, grid_size = 10, discount = 0.9):
    if memory.shape[0] < max_memory:
        mem_rows = memory.shape[0]
    else:
        mem_rows = max_memory
    n_actions = model.output.shape[-1]
    canvas_size = grid_size **2
    input_rows = min(mem_rows, batch_size)
    inputs = np.zeros((input_rows, canvas_size))
    print(inputs.shape)
    target = np.zeros((inputs.shape[0], n_actions))
    i = np.random.choice(np.arange(mem_rows), inputs.shape[0])
    idx = np.arange(mem_rows)
    # previous state, action, reward, state taken?
    print(memory.shape)
    state_vectors = memory[idx, :(memory.shape[-1] - 1)]
    game_status = memory[idx, -1]
    # update inputs
    for i in vals:
        inputs[i:i+1]
    inputs[i:i+1] = state_vectors[0]
    inputs.reshape(input_rows, canvas_size)
    targets[i] = model.predict(state_vectors[0])[:,0]
    Q_sa = np.max(model.predict(state_vectors[-1])[:,0])
    targets[np.where(game_status==1)] = state_vectors[np.where(game_status==1),2]
    targets[np.where(game_status==0)] = state_vectors[np.where(game_status==0),2] + discount*Q_sa
    return inputs, targets


def build_model(actions, grid_size = 10, hidden_size = 100):
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(actions.shape[0]))
    return(model)



def make_decision(epsilon, actions, model, initial_state):
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, actions.shape[0], size = 1)
    else:
        q = model.predict(initial_state)
        action = np.argmax(q[0])
    return action

def play_one_step(basket, grid_size,
                  epsilon, actions,
                  model, memory,
                  max_memory, discount,
                  batch_size, loss, n_win):
    init_state = get_current_state(basket, grid_size)
    action = make_decision(epsilon, actions, model, init_state)
    new_state, reward, game_status = one_step(basket, action, grid_size)
    if (reward == 1):
        n_win += 1
    # [input_tm1, action, reward, input_t], game_over
    #memory, experience, over, max_memory = 100
    memory = remember(memory, np.array([init_state, action, reward, new_state]), game_status)
    inputs, targets = get_one_batch(memory, model, batch_size, max_memory, grid_size, discount)
    loss += model.train_on_batch(inputs, targets)
    return memory, loss, update(basket, action, grid_size), game_status, n_win

def play_one_game(game_status, basket,
                  grid_size, epsilon,
                  actions, model,
                  memory, max_memory,
                  discount, batch_size,
                  loss, n_win):
    while not game_status:
        memory, loss, basket, game_status = play_one_step(basket, grid_size,
                                                          epsilon, actions,
                                                          model, memory,
                                                          max_memory, discount,
                                                          batch_size, loss,
                                                          n_win)
    return memory, loss, n_win

def play_for_ages(grid_size, epsilon,
                  actions, model,
                  memory, max_memory,
                  discount, batch_size, epochs):
    for e in range(epochs):
        loss = 0.
        basket = initialize(grid_size)
        n_win = 0
        game_status = False
        memory, loss, n_win = play_one_game(game_status, basket, grid_size, epsilon, actions, model, memory, max_memory, discount, batch_size, loss, n_win)
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, n_win))
        model.save_weights("model.h5", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(model.to_json(), outfile)
        return loss, memory, n_win




if __name__ == "__main__":
    epsilon = 0.1
    actions = np.arange(3) - 1
    epochs = 1000
    max_memory = 500
    batch_size = 50
    grid_size = 10
    discount = 0.9
    model = build_model(np.arange(3), 10, 100)
    model.compile(sgd(lr=0.2), "mse")
    memory = np.zeros(actions.shape[0] + 2).reshape(1, actions.shape[0] + 2)
    loss, memory, n_win = play_for_ages(grid_size, epsilon,
                                        actions, model,
                                        memory, max_memory,
                                        discount, batch_size,
                                        epochs)


#    # parameters
#    epsilon = .1  # exploration
#    actions = np.arange(3) - 1
#    epoch = 1000
#    max_memory = 500
#    hidden_size = 100
#    batch_size = 50
#    grid_size = 10
#
#    model = Sequential()
#    model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
#    model.add(Dense(hidden_size, activation='relu'))
#    model.add(Dense(num_actions))
#    model.compile(sgd(lr=.2), "mse")
#
#    # If you want to continue training from a previous model, just uncomment the line bellow
#    # model.load_weights("model.h5")
#
#    # Define environment/game
#
#    # Initialize memory
#    mem = np.zeros(actions.s).reshape(1,5)
#
#    # Train
#    win_cnt = 0
#    for e in range(epoch):
#        loss = 0.
#        fruit_basket = initialize(grid_size)
#        game_over = False
#        # get initial input
#        input_t = get_current_state(fruit_basket, grid_size)
#        print(input_t)
#
#
#        while not game_over:
#            input_tm1 = input_t
#            # get next action
#            if np.random.rand() <= epsilon:
#                action = np.random.randint(0, num_actions, size=1)
#            else:
#                q = model.predict(input_tm1)
#                action = np.argmax(q[0])
#
#            # apply action, get rewards and new state
#            input_t, reward, game_over = env.act(action)
#            if reward == 1:
#                win_cnt += 1
#
#            # store experience
#            exp_replay.remember([input_tm1, action, reward, input_t], game_over)
#
#            # adapt model
#            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
#
#            loss += model.train_on_batch(inputs, targets)
#        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))
#
#    # Save trained model weights and architecture, this will be used by the visualization code
#    model.save_weights("model.h5", overwrite=True)
#    with open("model.json", "w") as outfile:
#        json.dump(model.to_json(), outfile)
