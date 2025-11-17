from .utils import chess_manager, GameContext
from chess import Move
import os
import sys
from pathlib import Path

from .engine_wrapper import EngineWrapper

_engine = None


def get_engine() -> EngineWrapper:
    """Get or initialize the engine singleton."""
    global _engine
    if _engine is None:
        # Find engine binary
        bot_dir = Path(__file__).parent.parent
        engine_dir = bot_dir / "engine" / "build"
        engine_path = engine_dir / "chess_engine"

        if not engine_path.exists():
            # Try building
            print(f"Engine not found at {engine_path}, please build it first.", file=sys.stderr)
            print(f"Run: cd {bot_dir / 'engine'} && bash build.sh", file=sys.stderr)
            raise RuntimeError(f"Engine binary not found: {engine_path}")

        # Find NNUE model (optional)
        nnue_model = os.getenv("NNUE_MODEL", "")
        if nnue_model and not Path(nnue_model).exists():
            nnue_model = ""

        _engine = EngineWrapper(str(engine_path), nnue_model if nnue_model else None)
        print(f"Engine initialized: {engine_path}", file=sys.stderr)
        if nnue_model:
            print(f"NNUE model: {nnue_model}", file=sys.stderr)

    return _engine


@chess_manager.entrypoint
def make_move(ctx: GameContext) -> Move:
    engine = get_engine()

    best_move, score, info_lines = engine.get_best_move(ctx.board, ctx.timeLeft)

    # Log the last info line for debugging
    for line in info_lines[-3:]:
        if "info" in line:
            print(line)

    # Report probabilities: engine gives us a single best move
    # Assign probability 1.0 to best move, 0.0 to others
    legal_moves = list(ctx.board.generate_legal_moves())
    move_probs = {m: 0.0 for m in legal_moves}
    if best_move in move_probs:
        move_probs[best_move] = 1.0
    else:
        # Fallback: the engine might report a move in different format
        for m in legal_moves:
            if m.uci() == best_move.uci():
                move_probs[m] = 1.0
                best_move = m
                break

    ctx.logProbabilities(move_probs)
    return best_move


@chess_manager.reset
def reset_game(ctx: GameContext):
    """Called when a new game starts."""
    engine = get_engine()
    engine.new_game()
