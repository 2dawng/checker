# Game logic module
import pygame
import time
import random
import math
import os
from typing import List, Tuple, Optional
from constants import *
from board import Board
from piece import Piece
from neural_net import checkers_nn
from nnue import checkers_nnue  # Import the NNUE engine

# Add debug logging to check if NNUE is loaded properly
print(f"NNUE model loaded: {hasattr(checkers_nnue, 'weights1')}")
if hasattr(checkers_nnue, 'weights1'):
    print(f"NNUE model input size: {checkers_nnue.input_size}, hidden size: {checkers_nnue.hidden_size}")

def log_move(game, piece, row, col, captured_pieces=None, start_pos=None):
    """Record moves in algebraic notation for both file logging and in-game display"""
    start_pos = start_pos or game.board.get_square_name(piece.row, piece.col)
    end_pos = game.board.get_square_name(row, col)

    # Check if this is a sequential capture by the same player
    last_was_capture = (len(game.move_log) > 0 and
                        len(game.move_log[-1]) > 0 and
                        'x' in game.move_log[-1][-1] and
                        game.last_capture_color == piece.color)

    # For file logging
    if piece.color == PLAYER1_COLOR:
        if captured_pieces and game.move_number > 0 and last_was_capture:
            # Sequential capture for Player 1 - newline and one tab
            file_text = f"\n\t{start_pos}-{end_pos}"
        else:
            # New turn - include turn number, only add newline if not first move
            file_text = f"{game.current_turn_number}.\t{start_pos}-{end_pos}" if game.move_number == 0 else f"\n{game.current_turn_number}.\t{start_pos}-{end_pos}"

        if captured_pieces:
            file_text += f"x{len(captured_pieces)}"
            game.last_capture_color = piece.color
        else:
            game.last_capture_color = None
        if piece.king and not piece.was_king:
            file_text += "K"
        game.log_file.write(file_text)

    else:  # Player 2's moves
        if captured_pieces and game.move_number > 0 and last_was_capture:
            # Sequential capture for Player 2 - newline and four tabs
            file_text = f"\n\t\t\t\t{start_pos}-{end_pos}"
        else:
            # Normal move - two tabs
            file_text = f"\t\t{start_pos}-{end_pos}"

        if captured_pieces:
            file_text += f"x{len(captured_pieces)}"
            game.last_capture_color = piece.color
        else:
            game.last_capture_color = None
        if piece.king and not piece.was_king:
            file_text += "K"
        game.log_file.write(file_text)

    # For in-game display
    move_text = f"{start_pos}-{end_pos}"
    if captured_pieces:
        move_text += f"x{len(captured_pieces)}"
    if piece.king and not piece.was_king:
        move_text += "K"

    # For in-game display, use current_turn_number consistently
    if piece.color == PLAYER1_COLOR:
        if not (captured_pieces and last_was_capture):
            display_text = f"{game.current_turn_number}. {move_text}"
        else:
            display_text = f" {move_text}"
    else:
        display_text = f" {move_text}"

    # Store move information
    if piece.color == PLAYER1_COLOR and not (captured_pieces and last_was_capture):
        game.move_log.append([display_text])  # Start new turn
    else:
        if game.move_log:  # Add to current turn
            game.move_log[-1].append(display_text)

    game.log_file.flush()
    game.move_number += 1

def draw_move_log(game):
    """Draw the move log in the side panel"""
    # Draw move log title
    font = pygame.font.Font(None, 36)
    title = font.render("Move Log", True, WHITE)
    game.win.blit(title, (BOARD_WIDTH + 20, 220))

    # Draw moves with smaller font
    font = pygame.font.Font(None, 20)  # Reduced font size
    y_pos = 270
    max_moves = 20  # Show more moves since they take less space

    # Calculate start index for display
    total_turns = len(game.move_log)
    start_idx = max(0, total_turns - max_moves)

    # Display moves
    for turn_moves in game.move_log[start_idx:]:
        # Combine all moves in the turn
        line = "".join(turn_moves)
        text = font.render(line, True, WHITE)
        game.win.blit(text, (BOARD_WIDTH + 20, y_pos))
        y_pos += 20  # Space between lines

def draw_valid_moves(game):
    """Draw visual indicators for valid moves and pieces that can capture"""
    # If captures are available
    if game.board.has_captures_available(game.turn):
        # If there's a capturing piece, only highlight that piece and its moves
        if game.capturing_piece:
            piece_x = game.capturing_piece.col * SQUARE_SIZE + SQUARE_SIZE//2
            piece_y = game.capturing_piece.row * SQUARE_SIZE + SQUARE_SIZE//2
            pygame.draw.circle(game.win, (0, 255, 255), (piece_x, piece_y), 10)

            if game.selected_piece == game.capturing_piece:
                moves = game.valid_moves
                for move in moves:
                    if len(moves[move]) > 0:  # Only show capture moves
                        row, col = move
                        pygame.draw.circle(game.win, (0, 255, 0),
                                           (col * SQUARE_SIZE + SQUARE_SIZE//2,
                                            row * SQUARE_SIZE + SQUARE_SIZE//2), 20)
        else:
            pieces_with_captures = game.board.get_pieces_with_captures(game.turn)
            # Draw indicators on all pieces that can capture
            for piece in pieces_with_captures:
                piece_x = piece.col * SQUARE_SIZE + SQUARE_SIZE//2
                piece_y = piece.row * SQUARE_SIZE + SQUARE_SIZE//2
                pygame.draw.circle(game.win, (0, 255, 255), (piece_x, piece_y), 10)

            # If a piece is selected, show only its capture destinations
            if game.selected_piece and game.selected_piece in pieces_with_captures:
                moves = game.valid_moves
                for move in moves:
                    if len(moves[move]) > 0:  # Only show capture moves
                        row, col = move
                        pygame.draw.circle(game.win, (0, 255, 0),
                                           (col * SQUARE_SIZE + SQUARE_SIZE//2,
                                            row * SQUARE_SIZE + SQUARE_SIZE//2), 20)

    # If no captures and a piece is selected, show its regular moves
    elif game.selected_piece:
        for move in game.valid_moves:
            row, col = move
            pygame.draw.circle(game.win, (0, 255, 0),
                               (col * SQUARE_SIZE + SQUARE_SIZE//2,
                                row * SQUARE_SIZE + SQUARE_SIZE//2), 15)

def evaluate_position(board):
    """Evaluate the current board position using neural network and heuristics"""
    # Use our neural network to evaluate the position
    return checkers_nn.evaluate(board)

def minimax_alpha_beta(board, depth, alpha, beta, maximizing_player, game):
    """Minimax algorithm with alpha-beta pruning for checker AI"""
    
    # Terminal conditions: reached depth limit or game over
    if depth == 0 or board.check_winner() is not None:
        return evaluate_position(board), None
    
    # Get the current player's color
    current_color = PLAYER2_COLOR if maximizing_player else PLAYER1_COLOR
    
    # Check if captures are available and enforce them
    has_captures = board.has_captures_available(current_color)
    
    if maximizing_player:  # Bot/White's turn - maximize score
        max_eval = float('-inf')
        best_move = None
        
        # Get all valid moves
        all_moves = board.get_all_moves(current_color)
        
        # Process only pieces with captures if captures are available
        if has_captures:
            pieces_with_captures = []
            for piece in all_moves:
                for move in all_moves[piece]:
                    if all_moves[piece][move]:  # If there are captures
                        pieces_with_captures.append(piece)
                        break
            
            move_sources = {p: all_moves[p] for p in pieces_with_captures}
        else:
            move_sources = all_moves
        
        # Try each possible move
        for piece in move_sources:
            for move, captures in move_sources[piece].items():
                # Skip non-capture moves if captures are available
                if has_captures and not captures:
                    continue
                
                # Create a copy of the board and apply the move
                temp_board = board.copy()
                temp_piece = temp_board.get_piece(piece.row, piece.col)
                temp_board.move(temp_piece, move[0], move[1])
                
                # Remove any captured pieces
                if captures:
                    pieces_to_remove = []
                    for captured in captures:
                        captured_piece = temp_board.get_piece(captured.row, captured.col)
                        if captured_piece != 0:
                            pieces_to_remove.append(captured_piece)
                    
                    if pieces_to_remove:
                        temp_board.remove(pieces_to_remove)
                
                # Check if additional captures are available after this move
                additional_captures = False
                if captures:
                    next_moves = temp_board.get_valid_moves(temp_piece)
                    additional_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                
                # If there are additional captures, continue with the same player
                if additional_captures:
                    eval_val, _ = minimax_alpha_beta(temp_board, depth-1, alpha, beta, maximizing_player, game)
                else:
                    # Otherwise switch to the opponent
                    eval_val, _ = minimax_alpha_beta(temp_board, depth-1, alpha, beta, False, game)
                
                # Update best move if this is better
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = (piece, move)
                
                # Alpha-beta pruning
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
        
        return max_eval, best_move
    
    else:  # Human/Black's turn - minimize score
        min_eval = float('inf')
        best_move = None
        
        # Get all valid moves
        all_moves = board.get_all_moves(current_color)
        
        # Process only pieces with captures if captures are available
        if has_captures:
            pieces_with_captures = []
            for piece in all_moves:
                for move in all_moves[piece]:
                    if all_moves[piece][move]:  # If there are captures
                        pieces_with_captures.append(piece)
                        break
            
            move_sources = {p: all_moves[p] for p in pieces_with_captures}
        else:
            move_sources = all_moves
        
        # Try each possible move
        for piece in move_sources:
            for move, captures in move_sources[piece].items():
                # Skip non-capture moves if captures are available
                if has_captures and not captures:
                    continue
                
                # Create a copy of the board and apply the move
                temp_board = board.copy()
                temp_piece = temp_board.get_piece(piece.row, piece.col)
                temp_board.move(temp_piece, move[0], move[1])
                
                # Remove any captured pieces
                if captures:
                    pieces_to_remove = []
                    for captured in captures:
                        captured_piece = temp_board.get_piece(captured.row, captured.col)
                        if captured_piece != 0:
                            pieces_to_remove.append(captured_piece)
                    
                    if pieces_to_remove:
                        temp_board.remove(pieces_to_remove)
                
                # Check if additional captures are available after this move
                additional_captures = False
                if captures:
                    next_moves = temp_board.get_valid_moves(temp_piece)
                    additional_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                
                # If there are additional captures, continue with the same player
                if additional_captures:
                    eval_val, _ = minimax_alpha_beta(temp_board, depth-1, alpha, beta, maximizing_player, game)
                else:
                    # Otherwise switch to the opponent
                    eval_val, _ = minimax_alpha_beta(temp_board, depth-1, alpha, beta, True, game)
                
                # Update best move if this is better
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = (piece, move)
                
                # Alpha-beta pruning
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
        
        return min_eval, best_move

def nnue_search(board, depth, alpha, beta, maximizing_player, game, old_features=None, parent_move=None):
    """
    Alpha-beta search with NNUE evaluation
    This is much faster than traditional minimax as it uses incremental updates
    """
    # Terminal conditions: reached depth limit or game over
    if depth == 0 or board.check_winner() is not None:
        return checkers_nnue.evaluate(board, old_features, parent_move), None
    
    # Get the current player's color
    current_color = PLAYER2_COLOR if maximizing_player else PLAYER1_COLOR
    
    # Check if captures are available and enforce them
    has_captures = board.has_captures_available(current_color)
    
    if maximizing_player:  # Bot/White's turn - maximize score
        max_eval = float('-inf')
        best_move = None
        
        # Get all valid moves
        all_moves = board.get_all_moves(current_color)
        
        # Process only pieces with captures if captures are available
        if has_captures:
            pieces_with_captures = []
            for piece in all_moves:
                for move in all_moves[piece]:
                    if all_moves[piece][move]:  # If there are captures
                        pieces_with_captures.append(piece)
                        break
            
            move_sources = {p: all_moves[p] for p in pieces_with_captures}
        else:
            move_sources = all_moves
        
        # If we're at the root, compute features for the board
        if old_features is None:
            current_features = checkers_nnue.board_to_features(board)
        else:
            current_features = old_features
        
        # Try each possible move
        for piece in move_sources:
            for move, captures in move_sources[piece].items():
                # Skip non-capture moves if captures are available
                if has_captures and not captures:
                    continue
                
                # Create a copy of the board and apply the move
                temp_board = board.copy()
                temp_piece = temp_board.get_piece(piece.row, piece.col)
                
                # The actual move
                move_tuple = (temp_piece, move)
                temp_board.move(temp_piece, move[0], move[1])
                
                # Remove any captured pieces
                if captures:
                    pieces_to_remove = []
                    for captured in captures:
                        captured_piece = temp_board.get_piece(captured.row, captured.col)
                        if captured_piece != 0:
                            pieces_to_remove.append(captured_piece)
                    
                    if pieces_to_remove:
                        temp_board.remove(pieces_to_remove)
                
                # Check if additional captures are available after this move
                additional_captures = False
                if captures:
                    next_moves = temp_board.get_valid_moves(temp_piece)
                    additional_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                
                # Update features incrementally
                new_features = checkers_nnue.incremental_update(temp_board, current_features, move_tuple)
                
                # If there are additional captures, continue with the same player
                if additional_captures:
                    eval_val, _ = nnue_search(temp_board, depth-1, alpha, beta, maximizing_player, game, new_features, move_tuple)
                else:
                    # Otherwise switch to the opponent
                    eval_val, _ = nnue_search(temp_board, depth-1, alpha, beta, False, game, new_features, move_tuple)
                
                # Update best move if this is better
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = (piece, move)
                
                # Alpha-beta pruning
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
        
        return max_eval, best_move
    
    else:  # Human/Black's turn - minimize score
        min_eval = float('inf')
        best_move = None
        
        # Get all valid moves
        all_moves = board.get_all_moves(current_color)
        
        # Process only pieces with captures if captures are available
        if has_captures:
            pieces_with_captures = []
            for piece in all_moves:
                for move in all_moves[piece]:
                    if all_moves[piece][move]:  # If there are captures
                        pieces_with_captures.append(piece)
                        break
            
            move_sources = {p: all_moves[p] for p in pieces_with_captures}
        else:
            move_sources = all_moves
        
        # If we're at the root, compute features for the board
        if old_features is None:
            current_features = checkers_nnue.board_to_features(board)
        else:
            current_features = old_features
        
        # Try each possible move
        for piece in move_sources:
            for move, captures in move_sources[piece].items():
                # Skip non-capture moves if captures are available
                if has_captures and not captures:
                    continue
                
                # Create a copy of the board and apply the move
                temp_board = board.copy()
                temp_piece = temp_board.get_piece(piece.row, piece.col)
                
                # The actual move
                move_tuple = (temp_piece, move)
                temp_board.move(temp_piece, move[0], move[1])
                
                # Remove any captured pieces
                if captures:
                    pieces_to_remove = []
                    for captured in captures:
                        captured_piece = temp_board.get_piece(captured.row, captured.col)
                        if captured_piece != 0:
                            pieces_to_remove.append(captured_piece)
                    
                    if pieces_to_remove:
                        temp_board.remove(pieces_to_remove)
                
                # Check if additional captures are available after this move
                additional_captures = False
                if captures:
                    next_moves = temp_board.get_valid_moves(temp_piece)
                    additional_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                
                # Update features incrementally
                new_features = checkers_nnue.incremental_update(temp_board, current_features, move_tuple)
                
                # If there are additional captures, continue with the same player
                if additional_captures:
                    eval_val, _ = nnue_search(temp_board, depth-1, alpha, beta, maximizing_player, game, new_features, move_tuple)
                else:
                    # Otherwise switch to the opponent
                    eval_val, _ = nnue_search(temp_board, depth-1, alpha, beta, True, game, new_features, move_tuple)
                
                # Update best move if this is better
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = (piece, move)
                
                # Alpha-beta pruning
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
        
        return min_eval, best_move

def find_best_move(game, depth=3):
    """Find the best move for the bot using NNUE search"""
    # Use NNUE search for better performance at higher depths
    _, best_move = nnue_search(
        game.board, depth, float('-inf'), float('inf'), True, game
    )
    return best_move

def bot_move(game):
    """Calculate and make bot's move - with NNUE evaluation"""
    try:
        print("Bot is thinking with NNUE evaluation...")
        if game.turn != PLAYER2_COLOR:
            game.bot_thinking = False
            return
            
        start_time = time.time()
        
        # If there are captures available, we must take one
        if game.board.has_captures_available(PLAYER2_COLOR):
            # Check if we're in the middle of a multi-capture sequence
            if game.waiting_for_continued_capture and game.capturing_piece:
                piece = game.capturing_piece
                moves = game.board.get_valid_moves(piece)
                capture_moves = {pos: captures for pos, captures in moves.items() if captures}
                
                if capture_moves:
                    # Find the best capture move
                    best_score = -float('inf')
                    best_move = None
                    for pos, captures in capture_moves.items():
                        # Make the move temporarily
                        old_row, old_col = piece.row, piece.col
                        game.board.move(piece, pos[0], pos[1])
                        game.board.remove(captures)
                        
                        # Evaluate the position
                        score = -checkers_nnue.evaluate(game.board)
                        
                        # Undo the move
                        piece.row, piece.col = old_row, old_col
                        game.board.board[old_row][old_col] = piece
                        game.board.board[pos[0]][pos[1]] = 0
                        for p in captures:
                            game.board.board[p.row][p.col] = p
                        
                        if score > best_score:
                            best_score = score
                            best_move = pos
                    
                    if best_move:
                        # Store original position before moving
                        start_row, start_col = piece.row, piece.col
                        start_pos = game.board.get_square_name(start_row, start_col)
                        
                        # Execute the best capture
                        captures = capture_moves[best_move]
                        game.board.move(piece, best_move[0], best_move[1])
                        game.board.remove(captures)
                        
                        # Record the move with original position
                        log_move(game, piece, best_move[0], best_move[1], captures, start_pos=start_pos)
                        
                        # Update visualization
                        game.last_moved_piece = piece
                        game.last_move_start = (start_row, start_col)  # Store the actual start position
                        game.last_captured_pieces = captures.copy() if captures else []
                        
                        # Check for more captures
                        next_moves = game.board.get_valid_moves(piece)
                        has_more_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                        
                        if has_more_captures:
                            # Continue the capture sequence
                            game.capturing_piece = piece
                            game.bot_thinking = False
                            game.waiting_for_continued_capture = True
                            return
                
                # No more captures in the sequence, change turn
                game.capturing_piece = None
                game.bot_thinking = False
                game.waiting_for_continued_capture = False
                game.change_turn()
                return
            
            # Not in a sequence yet - find a piece that can capture
            pieces_with_captures = game.board.get_pieces_with_captures(PLAYER2_COLOR)
            
            # If there's only one piece that can capture, and only one way to capture
            if len(pieces_with_captures) == 1:
                piece = pieces_with_captures[0]
                moves = game.board.get_valid_moves(piece)
                capture_moves = {pos: captures for pos, captures in moves.items() if captures}
                
                if len(capture_moves) == 1:
                    pos = list(capture_moves.keys())[0]
                    captures = list(capture_moves.values())[0]
                    
                    # Add a small delay to make it look like the bot is thinking
                    time.sleep(0.5)
                    
                    # Store original position before moving
                    start_row, start_col = piece.row, piece.col
                    start_pos = game.board.get_square_name(start_row, start_col)
                    
                    # Make the capture move
                    game.board.move(piece, pos[0], pos[1])
                    game.board.remove(captures)
                    
                    # Record the move with original position
                    log_move(game, piece, pos[0], pos[1], captures, start_pos=start_pos)
                    
                    # Update for visualization
                    game.last_moved_piece = piece
                    game.last_move_start = (start_row, start_col)  # Store the actual start position
                    game.last_captured_pieces = captures.copy() if captures else []
                    
                    # Check for additional captures
                    next_moves = game.board.get_valid_moves(piece)
                    has_more_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                    
                    if has_more_captures:
                        game.capturing_piece = piece
                        game.bot_thinking = False
                        game.waiting_for_continued_capture = True
                        return
                    
                    # If no more captures, change turn
                    game.bot_thinking = False
                    game.waiting_for_continued_capture = False
                    game.change_turn()
                    return
            
            # Multiple pieces can capture or multiple ways to capture
            # Let the NNUE decide which one is best
            best_score = -float('inf')
            best_move = None
            best_piece = None
            
            for piece in pieces_with_captures:
                moves = game.board.get_valid_moves(piece)
                for pos, captures in moves.items():
                    if captures:  # Only consider capture moves
                        # Make the move temporarily
                        old_row, old_col = piece.row, piece.col
                        game.board.move(piece, pos[0], pos[1])
                        game.board.remove(captures)
                        
                        # Evaluate the position with NNUE
                        score = -checkers_nnue.evaluate(game.board)
                        
                        # Undo the move
                        piece.row, piece.col = old_row, old_col
                        game.board.board[old_row][old_col] = piece
                        game.board.board[pos[0]][pos[1]] = 0
                        for p in captures:
                            game.board.board[p.row][p.col] = p
                        
                        if score > best_score:
                            best_score = score
                            best_move = pos
                            best_piece = piece
            
            if best_move and best_piece:
                # Store original position before moving
                start_row, start_col = best_piece.row, best_piece.col
                start_pos = game.board.get_square_name(start_row, start_col)
                
                # Make the best capture move
                captures = game.board.get_valid_moves(best_piece)[best_move]
                game.board.move(best_piece, best_move[0], best_move[1])
                game.board.remove(captures)
                
                # Record the move with original position
                log_move(game, best_piece, best_move[0], best_move[1], captures, start_pos=start_pos)
                
                # Update for visualization
                game.last_moved_piece = best_piece
                game.last_move_start = (start_row, start_col)  # Store the actual start position
                game.last_captured_pieces = captures.copy() if captures else []
                
                # Check for additional captures
                next_moves = game.board.get_valid_moves(best_piece)
                has_more_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                
                if has_more_captures:
                    game.capturing_piece = best_piece
                    game.bot_thinking = False
                    game.waiting_for_continued_capture = True
                    return
                
                # If no more captures, change turn
                game.capturing_piece = None
                game.bot_thinking = False
                game.waiting_for_continued_capture = False
                game.change_turn()
                return
        
        # No captures available, make a regular move using NNUE
        best_score = -float('inf')
        best_move = None
        best_piece = None
        
        # Get all pieces and their possible moves
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = game.board.get_piece(row, col)
                if piece != 0 and piece.color == PLAYER2_COLOR:
                    moves = game.board.get_valid_moves(piece)
                    for pos, captures in moves.items():
                        # Make the move temporarily
                        old_row, old_col = piece.row, piece.col
                        game.board.move(piece, pos[0], pos[1])
                        
                        # Evaluate the position with NNUE
                        score = -checkers_nnue.evaluate(game.board)
                        
                        # Undo the move
                        piece.row, piece.col = old_row, old_col
                        game.board.board[old_row][old_col] = piece
                        game.board.board[pos[0]][pos[1]] = 0
                        
                        if score > best_score:
                            best_score = score
                            best_move = pos
                            best_piece = piece
        
        if best_move and best_piece:
            # Add a small delay to make it look like the bot is thinking
            time.sleep(0.5)
            
            # Store original position before moving
            start_row, start_col = best_piece.row, best_piece.col
            start_pos = game.board.get_square_name(start_row, start_col)
            
            # Make the best move
            game.board.move(best_piece, best_move[0], best_move[1])
            
            # Record the move with original position
            log_move(game, best_piece, best_move[0], best_move[1], None, start_pos=start_pos)
            
            # Update for visualization
            game.last_moved_piece = best_piece
            game.last_move_start = (start_row, start_col)  # Store the actual start position
            game.last_captured_pieces = []
            
            # Change turn
            game.bot_thinking = False
            game.waiting_for_continued_capture = False
            game.change_turn()
            return
                
        # If we got here, the bot has no valid moves (stalemate)
        game.bot_thinking = False
        game.waiting_for_continued_capture = False
        print("Bot has no valid moves!")
        return
    
    except Exception as e:
        # Log any errors and cancel bot thinking
        print(f"Error in bot_move: {e}")
        game.bot_thinking = False
        game.waiting_for_continued_capture = False
