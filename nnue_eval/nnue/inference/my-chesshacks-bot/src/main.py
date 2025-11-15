from .utils import chess_manager, GameContext
from chess import Move
import time
import os
import sys
from pathlib import Path

# Load environment variables from .env.local
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / "devtools" / ".env.local"
if env_path.exists():
    load_dotenv(env_path)

# Import NNUE evaluator
try:
    from .nnue_eval import NNUE
    NNUE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: NNUE not available: {e}")
    print("Falling back to random moves. Make sure nnue_wrapper.so is built.")
    NNUE_AVAILABLE = False

# Global NNUE instance (initialized once)
_nnue = None

def get_nnue():
    """Get or initialize the NNUE evaluator."""
    global _nnue
    if _nnue is None and NNUE_AVAILABLE:
        try:
            # Get model path from environment variable or use default
            model_name = os.getenv("NNUE_MODEL", "../custom/src/nn-49c1193b131c.nnue")
            
            # Try to find the model file
            # Path structure: src/main.py -> src/ -> my-chesshacks-bot/ -> inference/
            current_file = Path(__file__)  # src/main.py
            bot_dir = current_file.parent.parent  # my-chesshacks-bot/
            inference_dir = bot_dir.parent  # inference/
            
            # Try multiple locations
            possible_paths = [
                Path(model_name) if Path(model_name).is_absolute() else None,
                inference_dir / model_name,  # Relative to inference/
                inference_dir / "models" / model_name,  # inference/models/
            ]
            
            model_path = None
            for path in possible_paths:
                if path and path.exists() and path.is_file():
                    model_path = path
                    break
            
            if model_path:
                # Initialize NNUE directly (output filtering handled by decorator)
                _nnue = NNUE(str(model_path))
                print(f"✓ NNUE initialized with model: {model_path}", file=sys.stderr)
            else:
                print(f"Warning: NNUE model not found. Tried:", file=sys.stderr)
                for path in possible_paths:
                    if path:
                        print(f"  - {path}", file=sys.stderr)
                print("Falling back to random moves.", file=sys.stderr)
                print("Set NNUE_MODEL environment variable to specify model file.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to initialize NNUE: {e}", file=sys.stderr)
            print("Falling back to random moves.", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
    return _nnue


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    # Removed verbose logging to reduce stdout/stderr noise
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # Try to use NNUE evaluation
    nnue = get_nnue()
    
    if nnue is not None:
        # Evaluate all legal moves and pick the best one
        move_scores = {}
        for move in legal_moves:
            # Make the move
            ctx.board.push(move)
            # Evaluate the resulting position
            score = nnue.evaluate_board(ctx.board)
            # Undo the move
            ctx.board.pop()
            move_scores[move] = score
        
        # Find the best move (highest score for white, lowest for black)
        if ctx.board.turn:  # White's turn
            best_move = max(legal_moves, key=lambda m: move_scores[m])
        else:  # Black's turn
            best_move = min(legal_moves, key=lambda m: move_scores[m])
        
        # Convert scores to probabilities (softmax-like)
        # For white: higher score = higher probability
        # For black: lower score (more negative) = higher probability
        if ctx.board.turn:
            # White: use scores directly
            max_score = max(move_scores.values())
            move_weights = {m: max(0, score - max_score + 100) for m, score in move_scores.items()}
        else:
            # Black: negate scores (lower is better)
            min_score = min(move_scores.values())
            move_weights = {m: max(0, (min_score - score) + 100) for m, score in move_scores.items()}
        
        total_weight = sum(move_weights.values())
        if total_weight > 0:
            move_probs = {move: weight / total_weight for move, weight in move_weights.items()}
        else:
            # Fallback to uniform
            move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        
        ctx.logProbabilities(move_probs)
        
        # Only log if score is non-zero (to reduce noise)
        if move_scores[best_move] != 0:
            print(f"Best move: {best_move.uci()} (score: {move_scores[best_move]} cp)")
        
        return best_move
    else:
        # Fallback to random if NNUE not available
        import random
        move_weights = [1.0] * len(legal_moves)
        total_weight = len(legal_moves)
        move_probs = {move: 1.0 / total_weight for move in legal_moves}
        ctx.logProbabilities(move_probs)
        return random.choice(legal_moves)


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    # NNUE doesn't need reset, but we can reinitialize if needed
    pass
