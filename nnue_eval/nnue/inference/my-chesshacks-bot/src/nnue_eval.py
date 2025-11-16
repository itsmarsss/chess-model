"""
NNUE Evaluation using Stockfish UCI

This module provides a Python interface to NNUE evaluation via Stockfish binary.
"""

import subprocess
import os
from pathlib import Path
from typing import Optional
import threading
import queue
import time


class NNUE:
    """NNUE evaluation engine wrapper using Stockfish."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the NNUE engine.
        
        Args:
            model_path: Path to the .nnue model file. If None, uses Stockfish's built-in network.
        """
        # Find Stockfish binary
        inference_dir = Path(__file__).parent.parent.parent  # inference/
        stockfish_paths = [
            inference_dir / "custom" / "src" / "stockfish",
            inference_dir / "stockfish-official" / "src" / "stockfish",
            inference_dir / "Stockfish" / "src" / "stockfish",
            Path("/usr/local/bin/stockfish"),
            Path("/opt/homebrew/bin/stockfish"),
            Path("stockfish"),
        ]
        
        self.stockfish_path = None
        for path in stockfish_paths:
            if path.exists() and (path.is_file() or path == Path("stockfish")):
                self.stockfish_path = path
                break
        
        if not self.stockfish_path:
            raise RuntimeError(
                f"Stockfish binary not found. Searched: {stockfish_paths}\n"
                f"Build it with: cd {inference_dir}/Stockfish/src && make -j build ARCH=apple-silicon"
            )
        
        # Start Stockfish process
        try:
            self.process = subprocess.Popen(
                [str(self.stockfish_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                bufsize=1,
                cwd=str(self.stockfish_path.parent)
            )
            
            # Create a queue for reading output in a separate thread
            self.output_queue = queue.Queue()
            self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
            self.reader_thread.start()
            
        except Exception as e:
            raise RuntimeError(f"Failed to start Stockfish: {e}")
        
        # Check if process started
        import time
        time.sleep(0.1)
        if self.process.poll() is not None:
            stdout_output = self.process.stdout.read() if self.process.stdout else ""
            raise RuntimeError(f"Stockfish process died immediately. output: {stdout_output}")
        
        # Load custom network if provided
        if model_path:
            if Path(model_path).parent == Path("."):
                # Search for model file
                bot_dir = Path(__file__).parent.parent
                possible_paths = [
                    inference_dir / "models" / model_path,
                    bot_dir / model_path,
                    Path(model_path),
                ]
                for p in possible_paths:
                    if p.exists() and p.is_file():
                        model_path = str(p.absolute())
                        break
            
            # Set the EvalFile option
            eval_file_cmd = f"setoption name EvalFile value {model_path}"
            print(f"[NNUE] Loading model: {model_path}", flush=True)
            self._send_command(eval_file_cmd)
        
        self._send_command("uci")
        response_lines = self._wait_for("uciok", timeout=10)
        
        # Check if uciok was received
        if not any("uciok" in line for line in response_lines):
            raise RuntimeError(
                f"Stockfish did not respond with uciok. Response: {response_lines[:10]}"
            )
        self._send_command("isready")
        self._wait_for("readyok")
        
        self._lock = threading.Lock()
        self._initialized = True
    
    def _read_output(self):
        """Thread function to read Stockfish output into a queue."""
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.output_queue.put(line.strip())
        except:
            pass
    
    def _send_command(self, command: str):
        """Send a command to Stockfish."""
        # Check if process is still alive
        if self.process.poll() is not None:
            raise RuntimeError(f"Stockfish process died (exit code: {self.process.returncode})")
        
        try:
            if self.process.stdin:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
        except (BrokenPipeError, IOError) as e:
            raise RuntimeError(f"Stockfish process died: {e}")
    
    def _read_line(self, timeout: float = 1.0) -> str:
        """Read a single line from Stockfish with timeout."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return ""
    
    def _wait_for(self, expected: str, timeout: int = 5) -> list[str]:
        """Wait for expected response from Stockfish."""
        lines = []
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            line = self._read_line(0.1)
            if line:
                lines.append(line)
                if expected in line:
                    break
        return lines
    
    def evaluate_fen(self, fen: str, depth: int = 20) -> int:
        """
        Evaluate a position from a FEN string using search.
        
        Args:
            fen: FEN string of the position
            depth: Search depth in plies (default: 20)
            
        Returns:
            Evaluation in centipawns (positive = white better, negative = black better)
        """
        if not self._initialized:
            raise RuntimeError("NNUE engine not initialized")
        
        with self._lock:
            # Set position
            self._send_command(f"position fen {fen}")
            
            # Use search instead of static eval
            self._send_command(f"go depth {depth}")
            
            # Read search output until we get bestmove
            eval_score = 0
            start_time = time.time()
            
            while time.time() - start_time < 30.0:  # Increased timeout for search
                line = self._read_line(0.1)
                if not line:
                    continue
                
                # Parse info lines with score
                # Format: "info depth 15 ... score cp 25 ..."
                if "info" in line and "score cp" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "cp" and i + 1 < len(parts):
                                eval_score = int(parts[i + 1])
                    except (ValueError, IndexError):
                        pass
                
                # Handle mate scores
                elif "info" in line and "score mate" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "mate" and i + 1 < len(parts):
                                mate_in = int(parts[i + 1])
                                # Convert mate to large score
                                eval_score = 10000 if mate_in > 0 else -10000
                    except (ValueError, IndexError):
                        pass
                
                # Stop when we get bestmove
                if "bestmove" in line:
                    break
            
            return eval_score
    
    def evaluate_board(self, board, depth: int = 20) -> int:
        """
        Evaluate a python-chess Board object.
        
        Args:
            board: python-chess Board object
            depth: Search depth in plies (default: 20)
            
        Returns:
            Evaluation in centipawns (positive = white better, negative = black better)
        """
        fen = board.fen()
        return self.evaluate_fen(fen, depth)
    
    def cleanup(self):
        """Clean up resources."""
        if self._initialized:
            try:
                self._send_command("quit")
                self.process.wait(timeout=2)
            except:
                self.process.kill()
            self._initialized = False
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
