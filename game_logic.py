# Game logic module
import pygame
import time
import random
import math
from typing import List, Tuple, Optional
from constants import *
from board import Board
from piece import Piece

def log_move(game, piece, row, col, captured_pieces=None):
    """Record moves in algebraic notation for both file logging and in-game display"""
    start_pos = game.board.get_square_name(piece.row, piece.col)
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
    """Evaluate the current board position from Player2's perspective (white/bot)"""
    score = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board.get_piece(row, col)
            if piece != 0:
                # Base value: kings are worth more than regular pieces
                value = 5 if piece.king else 1
                
                # Position bonuses
                center_bonus = 0.1 * (4 - abs(col - 3.5))  # Preference for center columns
                
                # Forward progress bonus for non-kings
                if not piece.king:
                    if piece.color == PLAYER2_COLOR:
                        # White/bot pieces get bonus for advancing toward king row
                        progress_bonus = 0.2 * row
                    else:
                        # Black/human pieces get bonus for advancing toward king row
                        progress_bonus = 0.2 * (BOARD_SIZE - 1 - row)
                else:
                    progress_bonus = 0
                
                # Edge penalty (kings near edge are less mobile)
                edge_penalty = 0
                if piece.king:
                    if col == 0 or col == BOARD_SIZE - 1:
                        edge_penalty = 0.3
                
                # Calculate total piece value with all bonuses/penalties
                piece_value = value + center_bonus + progress_bonus - edge_penalty
                
                # Add to score (positive for bot/white, negative for human/black)
                if piece.color == PLAYER2_COLOR:
                    score += piece_value
                else:
                    score -= piece_value
    
    return score

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

def find_best_move(game, depth=3):
    """Find the best move for the bot using minimax"""
    _, best_move = minimax_alpha_beta(
        game.board, depth, float('-inf'), float('inf'), True, game
    )
    return best_move

def bot_move(game):
    """Improved AI opponent move selection using minimax with alpha-beta pruning"""
    if not game.game_over and not game.bot_thinking:
        game.bot_thinking = True
        
        # For very simple positions or early game, we can use a lower depth
        # to improve performance
        total_pieces = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if game.board.get_piece(row, col) != 0:
                    total_pieces += 1
        
        # Adjust depth based on number of pieces (faster decisions with more pieces)
        if total_pieces > 16:
            depth = 3  # Early to mid game
        elif total_pieces > 8:
            depth = 4  # Mid to late game
        else:
            depth = 5  # End game, deeper search




        depth = 3 # For testing, keep it shallow




        # Find the best move using minimax
        best_move = find_best_move(game, depth)
        
        if best_move:
            piece, move = best_move
            row, col = move
            
            # Store the piece's previous state for logging
            was_king = piece.king
            piece.was_king = was_king
            
            # Get valid moves and captures
            moves = game.board.get_valid_moves(piece)
            captured_pieces = moves.get(move, [])
            
            # Track original starting position - store this before any moves in this turn
            original_start = (piece.row, piece.col)
            game.last_move_start = original_start
            
            # Track captured pieces
            game.last_captured_pieces = captured_pieces.copy() if captured_pieces else []
            
            # Update draw detection variables
            # Record board position before the move
            current_state = game.encode_board_state()
            game.position_history.append(current_state)
            
            # If we captured pieces, reset the no-capture counter
            if captured_pieces:
                game.moves_without_capture = 0
            else:
                game.moves_without_capture += 1
                
            # If we moved a non-king piece, reset the no-pawn-move counter
            if not piece.king:
                game.moves_without_pawn_move = 0
            else:
                game.moves_without_pawn_move += 1
            
            # Log the move before making it
            if captured_pieces:
                log_move(game, piece, row, col, captured_pieces)
            else:
                log_move(game, piece, row, col)
            
            # Make the move
            game.board.move(piece, row, col)
            
            # Update animation
            game.last_moved_piece = piece
            game.animation_start_time = time.time()
            
            # Handle captures
            if captured_pieces:
                game.board.remove(captured_pieces)
                
                # Check for additional captures
                next_moves = game.board.get_valid_moves(piece)
                while any(len(next_moves[m]) > 0 for m in next_moves):
                    # Find best next capture
                    follow_up_move = find_best_move(game, 2)  # Use lower depth for sequential captures
                    
                    if not follow_up_move:
                        break
                        
                    follow_piece, follow_pos = follow_up_move
                    
                    # Update draw detection for the multi-capture
                    # Record board position before the follow-up move
                    current_state = game.encode_board_state()
                    game.position_history.append(current_state)
                    # Reset capture counter since we're making another capture
                    game.moves_without_capture = 0
                    # Update pawn move counter based on piece type
                    if not follow_piece.king:
                        game.moves_without_pawn_move = 0
                    
                    # Keep the original starting position from the first move
                    # game.last_move_start stays the same for the whole sequence
                    
                    # Add new captured pieces to the list
                    additional_captures = next_moves[follow_pos]
                    if additional_captures:
                        game.last_captured_pieces.extend(additional_captures)
                    
                    # Log and make the additional capture
                    log_move(game, follow_piece, follow_pos[0], follow_pos[1], next_moves[follow_pos])
                    game.board.move(follow_piece, follow_pos[0], follow_pos[1])
                    
                    # Update animation for multi-capture moves
                    game.last_moved_piece = follow_piece
                    game.animation_start_time = time.time()
                    
                    # Remove captured pieces
                    game.board.remove(next_moves[follow_pos])
                    
                    # Update next possible moves
                    next_moves = game.board.get_valid_moves(follow_piece)
            
            # Change turn after all moves are complete
            game.change_turn()
            
        game.bot_thinking = False
