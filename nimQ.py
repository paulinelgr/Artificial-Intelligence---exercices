import numpy as np
import random

NB_STICKS = 21
epsilon = 0.05
lr = 0.1
gamma = 0.995

Qtable = np.zeros((NB_STICKS,3), dtype=float)

################################################################

def playGame():
    sticks, player = NB_STICKS, 0
    while sticks > 0:
        choice = random.randint(1, min(sticks,3))
        print ("Player ", player, ":", sticks, "-", choice, "->", sticks-choice)
        sticks -= choice
        player = 1 - player
    print ("Player ", player, "won the game")

print ("Generate Random Game")
playGame();
input("Press Enter to continue...")

################################################################


def bestChoice(sticks):
    return np.argmax(Qtable[sticks-1])+1

################################################################

def generatePredictedGame(epsilon):
    sticks, player = NB_STICKS, random.randint(0,1)
    states = np.array([],dtype=int)
    choices = np.array([],dtype=int)

    while sticks > 0:
        choice = min ( bestChoice(sticks), sticks)
        if random.uniform(0, 1) < epsilon :
        	choice = random.randint(1, min(sticks,3))
        states = np.append (states, sticks-1)  # States start at 0
        choices = np.append (choices, choice-1)
        sticks -= choice
        player = 1 - player

    won_or_lost = 0;   # lost

    # updateQtable
    for i in (reversed(range(len(states)))):
        reward = 2.* won_or_lost - 1  # 1 or -1
        state = states[i]
        action = choices[i]
        new_state = states[i+1] if i+1 < len(states) else 0
        #print (state+1, action+1, new_state, reward)

        Qtable[state, action] = Qtable[state, action] + lr * (reward + gamma * np.max(Qtable[new_state, :]) - Qtable[state, action])

        won_or_lost = 1 - won_or_lost

################################################################

for i in range(500):
    generatePredictedGame(epsilon)

    if i % 100 == 0:
    	print (np.abs(np.round(Qtable)))
    	#print (np.argmax (Qtable, axis = 1) + 1)
    	for j in range(2,NB_STICKS+1):
        	print("       ", j, " : ", bestChoice(j))




