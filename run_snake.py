import numpy as np
import math
import random
import time
import turtle
from deap import base

class DisplayGame:
    def __init__(self, XSIZE, YSIZE):
        # SCREEN
        self.win = turtle.Screen()
        self.win.title("EVCO Snake game")
        self.win.bgcolor("grey")
        self.win.setup(width=(XSIZE*20)+40,height=(YSIZE*20)+40)
        #self.win.screensize((XSIZE*20)+20,(YSIZE*20)+20)
        self.win.tracer(0)

        #Snake Head
        self.head = turtle.Turtle()
        self.head.shape("square")
        self.head.color("green")

        # Snake food
        self.food = turtle.Turtle()
        self.food.shape("circle")
        self.food.color("yellow")
        self.food.penup()
        self.food.shapesize(0.55, 0.55)
        self.segments = []

    def reset(self, snake):
        for i in range(len(self.segments)):
            s = self.segments.pop()
            s.reset()
            s.hideturtle()
            del s
        self.head.penup()
        self.food.goto(-500, -500)
        self.head.goto(-500, -500)
        for i in range(len(snake)-1):
            self.add_snake_segment()
        self.update_segment_positions(snake)
       
    def update_food(self,new_food):
        self.food.goto(((new_food[1]-9)*20)+20, (((9-new_food[0])*20)-10)-20)
        
    def update_segment_positions(self, snake):
        self.head.goto(((snake[0][1]-9)*20)+20, (((9-snake[0][0])*20)-10)-20)
        for i in range(len(self.segments)):
            self.segments[i].goto(((snake[i+1][1]-9)*20)+20, (((9-snake[i+1][0])*20)-10)-20)

    def add_snake_segment(self):
        self.new_segment = turtle.Turtle()
        self.new_segment.speed(0)
        self.new_segment.shape("square")
        self.new_segment.color('black')
        self.new_segment.penup()
        self.segments.append(self.new_segment)


class foodPlacer:
    def __init__(self, _XSIZE, _YSIZE):
        self.XSIZE = _XSIZE
        self.YSIZE = _YSIZE
        #self.reset()

    def conv_index_to_coord(self, index):
        x = index % (XSIZE-2)
        y = index // (YSIZE-2)

        return [x+1,y+1]

    def get_food_placement_2out(self, network_food, input):
        out = network_food.feedForward(input)

        x = round(out[0] * (XSIZE - 1))
        y = round(out[1] * (YSIZE - 1))       
        
        valid = not snake_game.is_in_snake([x,y])

        return [x,y], valid


    def get_food_placement(self, network_food, input):
        out = network_food.feedForward(input)
        index = list(range(0,len(out)))

        lst = [i for val, i in sorted(zip(out, index))]
        
        valid = False
        while not valid:            
            next_best = self.conv_index_to_coord(lst.pop())
            valid = not snake_game.is_in_snake(next_best)

        return next_best


class snake:
    def __init__(self, _XSIZE, _YSIZE):
        self.XSIZE = _XSIZE
        self.YSIZE = _YSIZE
        self.reset()

    def reset(self):
        self.snake = [[8,10], [8,9], [8,8], [8,7], [8,6], [8,5], [8,4], [8,3], [8,2], [8,1],[8,0] ]# Initial snake co-ordinates [ypos,xpos]    
        self.food = self.place_food()
        self.ahead = []
        self.snake_direction = "right"

    def place_food(self):
        self.food = [random.randint(1, (YSIZE-2)), random.randint(1, (XSIZE-2))]
        while (self.food in self.snake):
            self.food = [random.randint(1, (YSIZE-2)), random.randint(1, (XSIZE-2))]
        return( self.food )
    
    def place_food_AI(self, food_location):
        self.food = food_location

    
    def is_in_snake(self, coord):
        return coord in self.snake


    def update_snake_position(self):
        self.snake.insert(0, [self.snake[0][0] + (self.snake_direction == "down" and 1) + (self.snake_direction == "up" and -1), self.snake[0][1] + (self.snake_direction == "left" and -1) + (self.snake_direction == "right" and 1)])

    def food_eaten(self):
        if self.snake[0] == self.food:    # When snake eats the food
            return True
        else:    
            last = self.snake.pop()  # [1] If it does not eat the food, it moves forward and so last tail item is removed
            return False
            
    def snake_turns_into_self(self):
        if self.snake[0] in self.snake[1:]:
            return True
        else:
            return False

    def snake_hit_wall(self):
        if self.snake[0][0] == 0 or self.snake[0][0] == (YSIZE-1) or self.snake[0][1] == 0 or self.snake[0][1] == (XSIZE-1):
            return True
        else:
            return False



    def get_board_state(self):
        board = []
        for i in range(0, YSIZE, 1):
            for j in range(0, XSIZE, 1):
                board.append([i,j] in self.snake[0:])
        return board


    # Example sensing functions

    def sense_same_food_y(self):
        if self.food[0] == self.snake[0][0]:
            return 1.
        else:
            return 0.

    def sense_same_food_x(self):
        if self.food[1] == self.snake[0][1]:
            return 1.
        else:
            return 0.


    def sense_same_food_y_dir(self):
        if self.food[0] == self.snake[0][0]:
            if self.food[1] > self.snake[0][1]:
                return 1.
            else:
                return -1.
        else:
            return 0.

    def sense_same_food_x_dir(self):
        if self.food[1] == self.snake[0][1]:
            if self.food[0] > self.snake[0][0]:
                return 1.
            else:
                return -1.
        else:
            return 0.

            
    def sense_same_food_x_left(self):
        if self.food[1] == self.snake[0][1]:
            if self.food[0] == self.snake[0][0]:
                return 1.
            else:
                return -1.
        else:
            return 0.


    def sense_food_left(self):
        if self.food[0] < self.snake[0][0]:
            return 1.
        else:
            return 0.
            
    def sense_food_right(self):
        if self.food[0] > self.snake[0][0]:
            return 1.
        else:
            return 0.

    def sense_food_up(self):        
        if self.food[1] > self.snake[0][1]:
            return 1.
        else:
            return 0.

    def sense_food_down(self):        
        if self.food[1] > self.snake[0][1]:
            return 1.
        else:
            return 0.

    def sense_block_left(self):      
        x = self.snake[0][0] - 1
        y = self.snake[0][1]

        if x == 0 or ([x,y] in self.snake[1:]):
            return 1.
        else:
            return 0.
            
    def sense_block_right(self):      
        x = self.snake[0][0] + 1
        y = self.snake[0][1]

        if x == (XSIZE-1) or ([x,y] in self.snake[1:]):
            return 1.
        else:
            return 0.
            
    def sense_block_up(self):      
        x = self.snake[0][0]
        y = self.snake[0][1] + 1

        if y == (YSIZE-1) or ([x,y] in self.snake[1:]):
            return 1.
        else:
            return 0.
            
    def sense_block_down(self):      
        x = self.snake[0][0]
        y = self.snake[0][1] - 1

        if y == 0 or ([x,y] in self.snake[1:]):
            return 1.
        else:
            return 0.


    def sense_block_left_l2(self):      
        x = self.snake[0][0] - 1
        y = self.snake[0][1]

        if x == 0 or ([x,y] in self.snake[1:]):
            return 1.
        elif x -1 == 0 or ([x-1,y] in self.snake[1:]):
            return 0.5
        else:
            return 0.
            

    def sense_block_right_l2(self):      
        x = self.snake[0][0] + 1
        y = self.snake[0][1]

        if x == (XSIZE-1) or ([x,y] in self.snake[1:]):
            return 1.
        elif x +1 == (XSIZE-1) or ([x+1,y] in self.snake[1:]):
            return 0.5
        else:
            return 0.
            

    def sense_block_up_l2(self):      
        x = self.snake[0][0]
        y = self.snake[0][1] + 1

        if y == (YSIZE-1) or ([x,y] in self.snake[1:]):
            return 1.
        elif y +1 == (YSIZE-1) or ([x,y+1] in self.snake[1:]):
            return 0.5
        else:
            return 0.
            

    def sense_block_down_l2(self):      
        x = self.snake[0][0]
        y = self.snake[0][1] - 1

        if y == 0 or ([x,y] in self.snake[1:]):
            return 1.
        elif y -1 == 0 or ([x,y-1] in self.snake[1:]):
            return 0.5
        else:
            return 0.


    def sense_food_horz(self):
        if self.food[0] > self.snake[0][0]:
            return 1.
        elif self.food[0] < self.snake[0][0]:
            return - 1.
        else:
            return 0.


    def sense_food_vert(self):        
        if self.food[1] > self.snake[0][1]:
            return 1.
        elif self.food[1] < self.snake[0][1]:
            return - 1.
        else:
            return 0.


    def sense_adj_squares(self):
        offsets = [[0,1],[0,-1],[1,0],[1,1],[1,-1],[-1,0],[-1,1],[-1,-1]]
        blocked = []
        for off in offsets:
            x_off = self.snake[0][0] + off[0]
            y_off = self.snake[0][1] + off[1]

            cond = x_off == 0 or x_off == (XSIZE-1) or y_off == 0 or y_off == (YSIZE-1)
            
            if [x_off, y_off] in self.snake[1:]:
                cond2 = True
            else:
                cond2 = False

            blocked.append(int(cond and cond2))

        return blocked

    def sense_cardinal_adj(self):
        offsets = [[0,1],[0,-1],[1,0],[-1,0]]
        blocked = []
        for off in offsets:
            x_off = self.snake[0][0] + off[0]
            y_off = self.snake[0][1] + off[1]

            cond = x_off == 0 or x_off == (YSIZE-1) or y_off == 0 or y_off == (XSIZE-1)
            
            if [x_off, y_off] in self.snake[1:]:
                cond2 = True
            else:
                cond2 = False

            blocked.append(int(cond and cond2))

        return blocked

    def dist_to_food(self):
        max_dist = 30
        
        x_delta = abs(self.snake[0][0] - self.food[0])
        y_delta = abs(self.snake[0][1] - self.food[1])

        return (x_delta + y_delta) / max_dist

    def sense_snake_in_dir_up(self):
        x = self.snake[0][0]
        y = self.snake[0][1]
        dist_up = (YSIZE - y)

        for i in range(1, dist_up, 1):
            if [x,y+i] in self.snake[1:]:
                return 1.
        return 0
                
    def sense_snake_in_dir_down(self):
        x = self.snake[0][0]
        y = self.snake[0][1]
        dist_down = YSIZE - (YSIZE - y)
        
        for i in range(1, dist_down, 1):
            if [x,y-i] in self.snake[1:]:
                return 1.
        return 0
                
    def sense_snake_in_dir_left(self):
        x = self.snake[0][0]
        y = self.snake[0][1]
        dist_left = XSIZE - (XSIZE - x)
        
        for i in range(1, dist_left, 1):
            if [x-i,y] in self.snake[1:]:
                return 1.
        return 0
                
    def sense_snake_in_dir_right(self):
        x = self.snake[0][0]
        y = self.snake[0][1]
        dist_up = XSIZE - (XSIZE - x)

        for i in range(1, dist_up, 1):
            if [x+i,y] in self.snake[1:]:
                return 1.
        return 0

    def sense_x_position(self):
        return self.snake[0][0] / (XSIZE-1)

    def sense_y_position(self):
        return self.snake[0][1] / (YSIZE-1)


def get_input(input_code):
    if input_code == 0:
        return [snake_game.sense_same_food_x_dir(),snake_game.sense_same_food_y_dir()] # 2
    elif input_code == 1:
        return [snake_game.sense_food_up(),snake_game.sense_food_down(),snake_game.sense_food_right(),snake_game.sense_food_left()] # 4
    elif input_code == 2:
        return [snake_game.sense_block_up(),snake_game.sense_block_down(),snake_game.sense_block_left(),snake_game.sense_block_right()] # 4
    elif input_code == 3:
        return [snake_game.dist_to_food()] # 1
    elif input_code == 4:
        return [snake_game.sense_snake_in_dir_up(),snake_game.sense_snake_in_dir_down(),snake_game.sense_snake_in_dir_left(),snake_game.sense_snake_in_dir_right()] # 4
    elif input_code == 5:
        return [snake_game.sense_x_position(), snake_game.sense_y_position()] # 2
    else:
        return 0
        

def get_input_length(input_code):
    if input_code == 0:
        return 2
    elif input_code == 1:
        return 4
    elif input_code == 2:
        return 4
    elif input_code == 3:
        return 1
    elif input_code == 4:
        return 4
    elif input_code == 5:
        return 2
    else:
        return 0


def get_inputs_length(input_codes):    
    input_len = 0

    for code in input_codes:
        input_len += get_input_length(code)

    return input_len


def get_inputs(input_codes):
    inputs = []

    for code in input_codes:
        inputs += get_input(code)
    #inputs = same_food_lvl_dir + food_direction_indv + block_dir_idnv

    return inputs


def run_game(display,snake_game, food_agent, network, network_food, input_codes, headless):

    score = 0
    snake_game.reset()
    if not headless:
        display.reset(snake_game.snake)
        display.win.update()
    snake_game.place_food()
    game_over = False
    snake_direction = "right"
    total_moves = 0
    moves = 0
    max_moves = 200

    flag = True
    while not game_over and moves < max_moves:

        # ****YOUR AI BELOW HERE******************
        inputs = get_inputs(input_codes)
        out = network.feedForward(inputs)
        max_index = np.argmax(out)

        if max_index == 0:
            new_snake_direction = "up"
            snake_dirs = [1, 0, 0, 0]
        elif max_index == 1:
            new_snake_direction = "right"
            snake_dirs = [0, 1, 0, 0]
        elif max_index == 2:
            new_snake_direction = "down"
            snake_dirs = [0, 0, 1, 0]
        else:
            new_snake_direction = "left"
            snake_dirs = [0, 0, 0, 1]
        
        # ****YOUR AI ABOVE HERE******************
        snake_game.snake_direction = new_snake_direction
        snake_game.update_snake_position()
        moves += 1
        total_moves += 1

        # Check if food is eaten
        if snake_game.food_eaten():
            food_input = snake_game.get_board_state() + snake_dirs

            new_food = food_agent.get_food_placement(network_food, food_input)
            snake_game.place_food_AI(new_food)
            score += 1
            moves = 0
            if not headless: display.add_snake_segment()

        # Game over if the snake runs over itself
        if snake_game.snake_turns_into_self():
            game_over = True

        # Game over if the snake goes through a wall
        if snake_game.snake_hit_wall():
            game_over = True

        if not headless:       
            display.update_food(snake_game.food)
            display.update_segment_positions(snake_game.snake)
            display.win.update()
            time.sleep(0.01) # Change this to modify the speed the game runs at when displayed.

    #print("\nFINAL score - " + str(score))
    #print()
    return score,total_moves


class MLP(object):
    def __init__(self, numInput, numHidden1, numHidden2, numOutput):
        self.fitness = 0
        self.numInput = numInput + 1 # Add bias node from input to hidden layer 1 only
        self.numHidden1 = numHidden1 # Feel free to adapt the code to add more biases if you wish
        self.numHidden2 = numHidden2
        self.numOutput = numOutput

        self.w_i_h1 = np.random.randn(self.numHidden1, self.numInput) 
        self.w_h1_h2 = np.random.randn(self.numHidden2, self.numHidden1) 
        self.w_h2_o = np.random.randn(self.numOutput, self.numHidden2)

        self.ReLU = lambda x : max(0,x)

    def sigmoid(self,x):
        try:
            ans = (1 / (1 + math.exp(-x)))
        except OverflowError:
            ans = float('inf')
        return ans


class MLP(MLP):
    def feedForward(self, inputs):
        inputsBias = inputs[:]
        inputsBias.insert(len(inputs),1)             # Add bias input

        h1 = np.dot(self.w_i_h1, inputsBias)         # feed input to hidden layer 1
        h1 = [self.ReLU(x) for x in h1]              # Activate hidden layer1
        
        h2 = np.dot(self.w_h1_h2, h1)                 # feed layer 1 to hidden layer 2
        h2 = [self.ReLU(x) for x in h2]              # Activate hidden layer 2

        output = np.dot(self.w_h2_o, h2)             # feed to output layer
        output = [self.sigmoid(x) for x in output]   # Activate output layer
        return output

#class MLP(MLP):
    
    def getWeightsLinear(self):
        flat_w_i_h1 = list(self.w_i_h1.flatten())
        flat_w_h1_h2 = list(self.w_h1_h2.flatten())
        flat_w_h2_o = list(self.w_h2_o.flatten())
        return( flat_w_i_h1 + flat_w_h1_h2 + flat_w_h2_o )

    def setWeightsLinear(self, Wgenome):
        numWeights_I_H1 = self.numHidden1 * self.numInput
        numWeights_H1_H2 = self.numHidden2 * self.numHidden1
        #numWeights_H2_O = self.numOutput * self.numHidden2

        self.w_i_h1 = np.array(Wgenome[:numWeights_I_H1])
        self.w_i_h1 = self.w_i_h1.reshape((self.numHidden1, self.numInput))
        
        self.w_h1_h2 = np.array(Wgenome[numWeights_I_H1:(numWeights_H1_H2+numWeights_I_H1)])
        self.w_h1_h2 = self.w_h1_h2.reshape((self.numHidden2, self.numHidden1))

        self.w_h2_o = np.array(Wgenome[(numWeights_H1_H2+numWeights_I_H1):])
        self.w_h2_o = self.w_h2_o.reshape((self.numOutput, self.numHidden2))


def create_networks(input_size, layer1_size, layer2_size, layer1_size_food, layer2_size_food, ):
    net_inputs = input_size
    net_layer1 = layer1_size
    net_layer2 = layer2_size
    net_out = 4

    food_net_inputs = 260
    food_net_layer1 = layer1_size_food
    food_net_layer2 = layer2_size_food
    food_net_out = 196

    ind_size = ((net_inputs+1) * net_layer1) + (net_layer1 * net_layer2) + (net_layer2 * net_out)

    ind_size_food = ((food_net_inputs+1) * food_net_layer1) + (food_net_layer1 * food_net_layer2) + (food_net_layer2 * food_net_out)

    network = MLP(net_inputs, net_layer1, net_layer2, net_out)
    network_food = MLP(food_net_inputs, food_net_layer1, food_net_layer2, food_net_out)

    return network, network_food, ind_size, ind_size_food


def evaluate_snake(individual, network, network_food, food, input_codes, HEADLESS):
    network.setWeightsLinear(individual)   # Load the individual's weights into the neural network
    network_food.setWeightsLinear(food)
    score, moves = run_game(display, snake_game, food_agent, network, network_food, input_codes, HEADLESS)
    weight_shift = math.tanh(score / 20)
    #fitness = score
    #fitness = score + (score / moves)
    fitness = score + ((1-weight_shift)*(moves / 300)) 
    #fitness = score + (weight_shift*(score / moves)) + ((1-weight_shift)*(moves/30))
    return (fitness,), score


XSIZE = YSIZE = 16 # Number of grid cells in each direction (do not change this)
HEADLESS = False
if not HEADLESS:
    display = DisplayGame(XSIZE,YSIZE)
snake_game = snake(XSIZE,YSIZE)
food_agent = foodPlacer(XSIZE,YSIZE) 
toolbox = base.Toolbox()
toolbox.register("evaluate_snake", evaluate_snake)  


network, network_food, x, y = create_networks(6,10,10,8,6) 
b_snake = np.loadtxt("best_snake.txt")
b_food = np.loadtxt("best_food.txt")

toolbox.evaluate_snake(b_snake, network, network_food, b_food,[0,2], HEADLESS)