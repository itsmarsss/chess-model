import subprocess
import threading
import queue
import time
from pathlib import Path
from typing import Optional
import chess


class EngineWrapper:
    """UCI protocol wrapper for the C++ chess engine."""

    def __init__(self, engine_path: str, nnue_model_path: Optional[str] = None):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise RuntimeError(f"Engine binary not found: {engine_path}")

        self.process = subprocess.Popen(
            [str(self.engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._output_queue = queue.Queue()
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()
        self._lock = threading.Lock()

        # Wait briefly for process to start
        time.sleep(0.05)
        if self.process.poll() is not None:
            raise RuntimeError("Engine process died immediately")

        # Initialize UCI
        self._send("uci")
        self._wait_for("uciok", timeout=5)

        if nnue_model_path and Path(nnue_model_path).exists():
            self._send(f"setoption name EvalFile value {nnue_model_path}")

        self._send("setoption name Hash value 64")
        self._send("isready")
        self._wait_for("readyok", timeout=10)

    def _read_output(self):
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self._output_queue.put(line.strip())
        except Exception:
            pass

    def _send(self, command: str):
        if self.process.poll() is not None:
            raise RuntimeError(f"Engine process died (exit code: {self.process.returncode})")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_line(self, timeout: float = 1.0) -> str:
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return ""

    def _wait_for(self, expected: str, timeout: float = 5.0) -> list[str]:
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            line = self._read_line(0.1)
            if line:
                lines.append(line)
                if expected in line:
                    break
        return lines

    def get_best_move(self, board: chess.Board, time_left_ms: int) -> tuple[chess.Move, int, list[str]]:
        """
        Get the best move from the engine.

        Returns: (move, score_cp, info_lines)
        """
        with self._lock:
            fen = board.fen()
            self._send(f"position fen {fen}")

            # Allocate time: 1/20 of remaining, min 100ms, max 5000ms
            movetime = max(100, min(time_left_ms // 20, 5000))
            self._send(f"go movetime {movetime}")

            # Parse output until bestmove
            score = 0
            info_lines = []
            start = time.time()
            best_move_uci = None

            while time.time() - start < (movetime / 1000.0 + 5.0):
                line = self._read_line(0.1)
                if not line:
                    continue

                info_lines.append(line)

                if "info" in line and "score cp" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "cp" and i + 1 < len(parts):
                                score = int(parts[i + 1])
                    except (ValueError, IndexError):
                        pass
                elif "info" in line and "score mate" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "mate" and i + 1 < len(parts):
                                mate_in = int(parts[i + 1])
                                score = 10000 if mate_in > 0 else -10000
                    except (ValueError, IndexError):
                        pass
                elif "bestmove" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        best_move_uci = parts[1]
                    break

            if best_move_uci is None:
                raise RuntimeError("Engine did not return a bestmove")

            move = chess.Move.from_uci(best_move_uci)
            return move, score, info_lines

    def new_game(self):
        with self._lock:
            self._send("ucinewgame")
            self._send("isready")
            self._wait_for("readyok", timeout=5)

    def quit(self):
        try:
            self._send("quit")
            self.process.wait(timeout=2)
        except Exception:
            self.process.kill()

    def __del__(self):
        try:
            self.quit()
        except Exception:
            pass
