# pacman-ai-gameplay
This project implements a competitive 2-player game on an 8x8 board using Artificial Intelligence techniques. Players compete to collect coins placed randomly across the board, and the game employs both Probabilistic Minimax and Genetic Algorithms (GAs) to drive player decisions.

## Rules
- 2-player Pacman
- 8x8 board
- Players begin in opposite corners
- Each cell is either empty, or contains a coin (initialized at random)
- When a player enters a cell with a coin, it consumes it, and its score is increased by 1
- Game ends when there are no more coins
- Players take turns to move. Each player can move one cell up, down, left, or right. Players cannot simultaneously occupy the same cell.
- Each players has full observability of the board: they know where the opposite player and all the coins are.
- Each coin has a 50% chance of becoming ”transparent”: when transparent, it is not consumed. When transparent, it has a 50% chance of becoming solid again.
- Players get a bonus for number of successive coins consumed in a row. E.g., if a players consumes 3 coins in 3 consecutive moves, their score for those 3 coins is squared (9, instead of 3).

## Requirements
Python 3.8 or above.

## Setup and Execution
#### Clone the Repository:
```git clone https://your-repository-url```
#### Run the Game:
```python main.py```

## How It Works
Minimax: The game starts with the Minimax algorithm to decide each player's move based on a utility function that evaluates the game state.
Genetic Algorithm: After 70 moves, the game switches to a GA that evaluates sequences of moves to optimize the collection of coins.
Game Loop: The game checks after each move whether the end condition (no more coins) is met. If not, the next player makes a move. The cycle repeats until the game ends.

## Customizing the Game
You can alter the game settings such as board size, coin distribution probability, Minimax Depth, and the number of moves before switching to GA by modifying the corresponding constants in the main.py file.

-------
This AI-driven game showcases how different AI techniques can be used in competitive settings to improve strategy dynamically as the game progresses. The implementation of both Minimax for initial gameplay and a Genetic Algorithm for optimized strategies in later stages demonstrates a hybrid approach to game AI.
