import pygame
from typing import List, Tuple, Optional, Dict
from constants import *
from piece import Piece


class Board:
    def __init__(self):
        self.board = []
        self.create_board()

    def draw_squares(self, win):
        font = pygame.font.Font(None, 24)  # Font size

        # Draw the squares
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(win, color, (col * SQUARE_SIZE,
                                 row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

                # Draw indicators only for the bottom row and leftmost column
                if row == BOARD_SIZE - 1:  # Bottom row
                    # Draw column letter (uppercase) in bottom right
                    letter = chr(ord('A') + col)  # Uppercase letter
                    indicator_color = DARK_SQUARE if color == LIGHT_SQUARE else LIGHT_SQUARE
                    text = font.render(letter, True, indicator_color)
                    text_rect = text.get_rect(
                        bottomright=(col * SQUARE_SIZE + SQUARE_SIZE - 5, row * SQUARE_SIZE + SQUARE_SIZE - 5))
                    win.blit(text, text_rect)

                if col == 0:  # Leftmost column
                    # Draw row number in top left
                    number = str(BOARD_SIZE - row)
                    color = LIGHT_SQUARE if (
                        row + col) % 2 == 0 else DARK_SQUARE
                    indicator_color = DARK_SQUARE if color == LIGHT_SQUARE else LIGHT_SQUARE
                    text = font.render(number, True, indicator_color)
                    text_rect = text.get_rect(
                        topleft=(col * SQUARE_SIZE + 5, row * SQUARE_SIZE + 5))
                    win.blit(text, text_rect)

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

    def get_valid_moves(self, piece):
        moves = {}
        left = piece.col - 1
        right = piece.col + 1
        row = piece.row

        # For kings, check all directions at once
        if piece.king:
            moves.update(self._traverse_left(
                row - 1, max(row - 3, -1), -1, piece.color, left, piece=piece))
            moves.update(self._traverse_right(
                row - 1, max(row - 3, -1), -1, piece.color, right, piece=piece))
            moves.update(self._traverse_left(
                row + 1, min(row + 3, BOARD_SIZE), 1, piece.color, left, piece=piece))
            moves.update(self._traverse_right(
                row + 1, min(row + 3, BOARD_SIZE), 1, piece.color, right, piece=piece))
        else:
            if piece.color == PLAYER1_COLOR:
                moves.update(self._traverse_left(
                    row - 1, max(row - 3, -1), -1, piece.color, left, piece=piece))
                moves.update(self._traverse_right(
                    row - 1, max(row - 3, -1), -1, piece.color, right, piece=piece))
            if piece.color == PLAYER2_COLOR:
                moves.update(self._traverse_left(
                    row + 1, min(row + 3, BOARD_SIZE), 1, piece.color, left, piece=piece))
                moves.update(self._traverse_right(
                    row + 1, min(row + 3, BOARD_SIZE), 1, piece.color, right, piece=piece))

        return moves

    def _traverse_left(self, start: int, stop: int, step: int, color: Tuple[int, int, int], left: int, piece=None, skipped=None, visited=None) -> Dict[Tuple[int, int], List[Piece]]:
        moves = {}
        last = []
        if skipped is None:
            skipped = []
        if visited is None:
            visited = set()

        for r in range(start, stop, step):
            if left < 0:
                break

            pos = (r, left)
            if pos in visited:
                break
            visited.add(pos)

            current = self.get_piece(r, left)
            if current == 0:
                if skipped and not last:
                    break
                elif skipped:
                    moves[pos] = last + skipped
                else:
                    moves[pos] = last

                if last:
                    # For kings, check in all four directions after a capture
                    if piece and piece.king:
                        new_skipped = last + skipped
                        # Check forward-left
                        if r + 1 < BOARD_SIZE and left - 1 >= 0:
                            moves.update(self._traverse_left(
                                r + 1, min(r + 3, BOARD_SIZE), 1, color, left - 1, piece=piece, skipped=new_skipped, visited=visited.copy()))
                        # Check forward-right
                        if r + 1 < BOARD_SIZE and left + 1 < BOARD_SIZE:
                            moves.update(self._traverse_right(
                                r + 1, min(r + 3, BOARD_SIZE), 1, color, left + 1, piece=piece, skipped=new_skipped, visited=visited.copy()))
                        # Check backward-left
                        if r - 1 >= 0 and left - 1 >= 0:
                            moves.update(self._traverse_left(
                                r - 1, max(r - 3, -1), -1, color, left - 1, piece=piece, skipped=new_skipped, visited=visited.copy()))
                        # Check backward-right
                        if r - 1 >= 0 and left + 1 < BOARD_SIZE:
                            moves.update(self._traverse_right(
                                r - 1, max(r - 3, -1), -1, color, left + 1, piece=piece, skipped=new_skipped, visited=visited.copy()))
                    else:
                        # For regular pieces, only check in the current direction
                        if step == -1:
                            row = max(r - 3, -1)
                        else:
                            row = min(r + 3, BOARD_SIZE)
                        if r + step >= 0 and r + step < BOARD_SIZE:
                            if left - 1 >= 0:
                                moves.update(self._traverse_left(
                                    r + step, row, step, color, left - 1, piece=piece, skipped=last + skipped, visited=visited.copy()))
                            if left + 1 < BOARD_SIZE:
                                moves.update(self._traverse_right(
                                    r + step, row, step, color, left + 1, piece=piece, skipped=last + skipped, visited=visited.copy()))
                break
            elif current.color == color:
                break
            else:
                last = [current]

            left -= 1

        return moves

    def _traverse_right(self, start: int, stop: int, step: int, color: Tuple[int, int, int], right: int, piece=None, skipped=None, visited=None) -> Dict[Tuple[int, int], List[Piece]]:
        moves = {}
        last = []
        if skipped is None:
            skipped = []
        if visited is None:
            visited = set()

        for r in range(start, stop, step):
            if right >= BOARD_SIZE:
                break

            pos = (r, right)
            if pos in visited:
                break
            visited.add(pos)

            current = self.get_piece(r, right)
            if current == 0:
                if skipped and not last:
                    break
                elif skipped:
                    moves[pos] = last + skipped
                else:
                    moves[pos] = last

                if last:
                    # For kings, check in all four directions after a capture
                    if piece and piece.king:
                        new_skipped = last + skipped
                        # Check forward-left
                        if r + 1 < BOARD_SIZE and right - 1 >= 0:
                            moves.update(self._traverse_left(
                                r + 1, min(r + 3, BOARD_SIZE), 1, color, right - 1, piece=piece, skipped=new_skipped, visited=visited.copy()))
                        # Check forward-right
                        if r + 1 < BOARD_SIZE and right + 1 < BOARD_SIZE:
                            moves.update(self._traverse_right(
                                r + 1, min(r + 3, BOARD_SIZE), 1, color, right + 1, piece=piece, skipped=new_skipped, visited=visited.copy()))
                        # Check backward-left
                        if r - 1 >= 0 and right - 1 >= 0:
                            moves.update(self._traverse_left(
                                r - 1, max(r - 3, -1), -1, color, right - 1, piece=piece, skipped=new_skipped, visited=visited.copy()))
                        # Check backward-right
                        if r - 1 >= 0 and right + 1 < BOARD_SIZE:
                            moves.update(self._traverse_right(
                                r - 1, max(r - 3, -1), -1, color, right + 1, piece=piece, skipped=new_skipped, visited=visited.copy()))
                    else:
                        # For regular pieces, only check in the current direction
                        if step == -1:
                            row = max(r - 3, -1)
                        else:
                            row = min(r + 3, BOARD_SIZE)
                        if r + step >= 0 and r + step < BOARD_SIZE:
                            if right - 1 >= 0:
                                moves.update(self._traverse_left(
                                    r + step, row, step, color, right - 1, piece=piece, skipped=last + skipped, visited=visited.copy()))
                            if right + 1 < BOARD_SIZE:
                                moves.update(self._traverse_right(
                                    r + step, row, step, color, right + 1, piece=piece, skipped=last + skipped, visited=visited.copy()))
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
