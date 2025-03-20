from typing import Tuple
from constants import SQUARE_SIZE


class Piece:
    def __init__(self, row: int, col: int, color: Tuple[int, int, int]):
        self.row = row
        self.col = col
        self.color = color
        self.king = False
        self.was_king = False  # Track previous king status
        self.x = 0
        self.y = 0
        self.calc_pos()

    def calc_pos(self):
        self.x = SQUARE_SIZE * self.col + SQUARE_SIZE // 2
        self.y = SQUARE_SIZE * self.row + SQUARE_SIZE // 2

    def make_king(self):
        self.was_king = self.king  # Store previous status before making king
        self.king = True
