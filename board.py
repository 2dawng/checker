import pygame
from typing import List, Tuple, Optional, Dict
from constants import *
from piece import Piece


class Board:
    def __init__(self):
        self.board = []
        self.create_board()

    def draw_squares(self, win):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(win, color, (col * SQUARE_SIZE,
                                 row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def create_board(self):
        for row in range(BOARD_SIZE):
            self.board.append([])
            for col in range(BOARD_SIZE):
                if col % 2 == ((row + 1) % 2):
                    if row < 3:
                        self.board[row].append(Piece(row, col, PLAYER2_COLOR))
                    elif row > 4:
                        self.board[row].append(Piece(row, col, PLAYER1_COLOR))
                    else:
                        self.board[row].append(0)
                else:
                    self.board[row].append(0)

    def draw(self, win):
        self.draw_squares(win)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board[row][col]
                if piece != 0:
                    # Draw piece shadow
                    pygame.draw.circle(
                        win, BLACK, (piece.x + 2, piece.y + 2), SQUARE_SIZE//2 - 12)
                    # Draw piece
                    pygame.draw.circle(
                        win, piece.color, (piece.x, piece.y), SQUARE_SIZE//2 - 12)
                    if piece.king:
                        pygame.draw.circle(
                            win, CROWN_COLOR, (piece.x, piece.y), SQUARE_SIZE//4)

    def move(self, piece: Piece, row: int, col: int) -> bool:
        self.board[piece.row][piece.col], self.board[row][col] = self.board[row][col], self.board[piece.row][piece.col]
        piece.row = row
        piece.col = col
        piece.calc_pos()

        if row == 0 and piece.color == PLAYER1_COLOR:
            piece.make_king()
        if row == BOARD_SIZE-1 and piece.color == PLAYER2_COLOR:
            piece.make_king()

        return True

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row][col]
        return None

    def get_valid_moves(self, piece: Piece) -> Dict[Tuple[int, int], List[Piece]]:
        moves = {}
        left = piece.col - 1
        right = piece.col + 1
        row = piece.row

        if piece.color == PLAYER1_COLOR or piece.king:
            moves.update(self._traverse_left(
                row - 1, max(row - 3, -1), -1, piece.color, left))
            moves.update(self._traverse_right(
                row - 1, max(row - 3, -1), -1, piece.color, right))
        if piece.color == PLAYER2_COLOR or piece.king:
            moves.update(self._traverse_left(
                row + 1, min(row + 3, BOARD_SIZE), 1, piece.color, left))
            moves.update(self._traverse_right(
                row + 1, min(row + 3, BOARD_SIZE), 1, piece.color, right))

        return moves

    def _traverse_left(self, start: int, stop: int, step: int, color: Tuple[int, int, int], left: int, skipped=[]) -> Dict[Tuple[int, int], List[Piece]]:
        moves = {}
        last = []

        for r in range(start, stop, step):
            if left < 0:
                break

            current = self.get_piece(r, left)
            if current == 0:
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, left)] = last + skipped
                else:
                    moves[(r, left)] = last

                if last:
                    if step == -1:
                        row = max(r - 3, -1)
                    else:
                        row = min(r + 3, BOARD_SIZE)
                    moves.update(self._traverse_left(
                        r + step, row, step, color, left - 1, skipped=last))
                    moves.update(self._traverse_right(
                        r + step, row, step, color, left + 1, skipped=last))
                break
            elif current.color == color:
                break
            else:
                last = [current]

            left -= 1

        return moves

    def _traverse_right(self, start: int, stop: int, step: int, color: Tuple[int, int, int], right: int, skipped=[]) -> Dict[Tuple[int, int], List[Piece]]:
        moves = {}
        last = []

        for r in range(start, stop, step):
            if right >= BOARD_SIZE:
                break

            current = self.get_piece(r, right)
            if current == 0:
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, right)] = last + skipped
                else:
                    moves[(r, right)] = last

                if last:
                    if step == -1:
                        row = max(r - 3, -1)
                    else:
                        row = min(r + 3, BOARD_SIZE)
                    moves.update(self._traverse_left(
                        r + step, row, step, color, right - 1, skipped=last))
                    moves.update(self._traverse_right(
                        r + step, row, step, color, right + 1, skipped=last))
                break
            elif current.color == color:
                break
            else:
                last = [current]

            right += 1

        return moves

    def remove(self, pieces: List[Piece]):
        for piece in pieces:
            self.board[piece.row][piece.col] = 0
