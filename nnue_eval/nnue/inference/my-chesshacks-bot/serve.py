from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import chess
import os
import sys
from pathlib import Path

# Load environment variables from .env.local if it exists
# MUST be done BEFORE importing src.main so the env var is available
try:
    from dotenv import load_dotenv
    
    # Try to find .env.local in multiple locations
    # serve.py can be run from my-chesshacks-bot/ or devtools/
    current_dir = Path(__file__).parent  # my-chesshacks-bot/
    devtools_dir = current_dir / 'devtools'
    
    # Try devtools/.env.local first (most common)
    env_paths = [
        devtools_dir / '.env.local',
        current_dir / '.env.local',
        Path('.env.local'),  # Current working directory
    ]
    
    loaded = False
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)  # override=True to ensure it takes effect
            loaded = True
            break
    
    if not loaded:
        # Silently continue if not found - it's optional
        pass
except ImportError:
    # Silently continue - python-dotenv is optional
    pass
except Exception:
    # Silently continue on any other error
    pass

# Import main module
from src.utils import chess_manager
from src import main

app = FastAPI()


@app.post("/")
async def root():
    return JSONResponse(content={"running": True})


@app.post("/move")
async def get_move(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(content={"error": "Invalid JSON", "details": str(e)}, status_code=400)

    if ("pgn" not in data or "timeleft" not in data):
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    pgn = data["pgn"]
    timeleft = data["timeleft"]  # in milliseconds

    chess_manager.set_context(pgn, timeleft)

    try:
        start_time = time.perf_counter()
        move, move_probs, logs = chess_manager.get_model_move()
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": "Bot raised an exception",
                "logs": None,
                "exception": str(e),
            },
            status_code=500,
        )

    # Confirm type of move_probs
    if not isinstance(move_probs, dict):
        return JSONResponse(content={"move": None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    for m, prob in move_probs.items():
        if not isinstance(m, chess.Move) or not isinstance(prob, float):
            return JSONResponse(content={"move": None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    # Translate move_probs to Dict[str, float]
    move_probs_dict = {move.uci(): prob for move, prob in move_probs.items()}

    return JSONResponse(content={"move": move.uci(), "error": None, "time_taken": time_taken, "move_probs": move_probs_dict, "logs": logs})

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chess bot server')
    parser.add_argument('--port', type=int, default=None, help='Server port (default: from SERVE_PORT env or 5058)')
    parser.add_argument('--model', type=str, default=None, help='NNUE model path (overrides NNUE_MODEL env var)')
    args = parser.parse_args()
    
    # Override NNUE_MODEL env var if --model is specified
    if args.model:
        os.environ['NNUE_MODEL'] = args.model
        print(f"Using model: {args.model}")
    
    port = args.port if args.port is not None else int(os.getenv("SERVE_PORT", "5058"))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
