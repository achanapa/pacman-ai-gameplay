# pacman-ai-gameplay

- 2-player Pacman
• 8x8 board
• Players begin in opposite corners
• Each cell is either empty, or contains a coin (initialized at random)
• When a player enters a cell with a coin, it consumes it, and its score is increased by 1
• Game ends when there are no more coins
• Players take turns to move. Each player can move one cell up, down, left,
or right. Players cannot simultaneously occupy the same cell.
• Each players has full observability of the board: they know where the opposite player and all the coins are.
• Each coin has a 50% chance of becoming ”transparent”: when transparent, it is not consumed. When transparent, it has a 50% chance of becoming solid again.
• Players get a bonus for number of successive coins consumed in a row. E.g., if a players consumes 3 coins in 3 consecutive moves, their score for those 3 coins is squared (9, instead of 3).
