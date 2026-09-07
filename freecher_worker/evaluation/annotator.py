"""Interactive terminal-based annotation interface for blind human evaluation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
from freecher_worker.utils.json_io import save_json
from .models import BlindEvaluationDocument, BlindEvaluationItem


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.S timestamp."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:04.1f}"


def preview_clip(
    video_path: Path,
    start: float,
    end: float,
    output_path: Optional[Path] = None,
    open_player: bool = True,
) -> Path:
    """Extract a fast preview clip using ffmpeg and optionally open in system media player."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        temp_dir = Path(tempfile.gettempdir()) / "freecher_previews"
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_dir / f"preview_{int(start)}_{int(end)}.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        str(output_path),
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        check=True,
        timeout=60,
    )

    if open_player:
        open_media_file(output_path)

    return output_path


def open_media_file(file_path: Path) -> None:
    """Open media file with platform default player."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(file_path)], check=False)
        elif sys.platform == "win32":
            os.startfile(str(file_path))  # type: ignore
        else:
            # Linux / WSL
            # Check for WSL
            if "microsoft" in Path("/proc/version").read_text().lower() if Path("/proc/version").exists() else False:
                # Try wslview or explorer.exe
                try:
                    subprocess.run(["wslview", str(file_path)], check=False)
                    return
                except FileNotFoundError:
                    pass
            subprocess.run(["xdg-open", str(file_path)], check=False)
    except Exception as exc:
        print(f"[preview] Unable to launch system media player: {exc}")


def run_terminal_annotator(
    eval_doc: BlindEvaluationDocument,
    eval_file_path: Path,
    video_path: Optional[Path] = None,
    re_label_all: bool = False,
) -> int:
    """Interactive CLI loop for blind evaluation of candidates.

    Returns the count of labeled candidates upon completion.
    """
    items = eval_doc.items
    total = len(items)
    if total == 0:
        print("Evaluation document contains no candidates.")
        return 0

    # Determine starting index (first unlabeled unless re_label_all is set)
    current_idx = 0
    if not re_label_all:
        for idx, item in enumerate(items):
            if item.human_score is None:
                current_idx = idx
                break
        else:
            print(f"All {total} candidates are already labeled.")
            ans = input("Re-evaluate all candidates from the start? [y/N]: ").strip().lower()
            if ans != "y":
                return eval_doc.update_labeled_count()

    print("\n" + "=" * 70)
    print("BLIND HIGHLIGHT EVALUATION SESSION")
    print(f"Candidate Set: {eval_doc.candidate_set_id}")
    print(f"Total candidates: {total}")
    print("Scale:")
    print("  0: Unusable (cut mid-sentence, no context, noise)")
    print("  1: Weak (dull, rambling, low interest)")
    print("  2: Acceptable (coherent thought, but mediocre hook/delivery)")
    print("  3: Good (clear hook, interesting content, strong candidate)")
    print("  4: Excellent (viral hook, punchline/payoff, standalone value)")
    print("Commands:")
    print("  [0-4] Score | [s] Skip | [b] Back | [p] Preview clip | [q] Quit & Save")
    print("=" * 70 + "\n")

    while 0 <= current_idx < total:
        item = items[current_idx]
        eval_doc.update_labeled_count()
        labeled = eval_doc.labeled_candidates

        print(f"\n--- [{current_idx + 1}/{total}] (Labeled: {labeled}/{total}) ---")
        print(f"Candidate ID: {item.candidate_id}")
        print(f"Timing:       {format_timestamp(item.start)} -> {format_timestamp(item.end)} ({item.duration:.1f}s)")
        print(f"Text:         \"{item.text}\"")

        if item.human_score is not None:
            print(f"Current Rating: score={item.human_score}, publishable={item.publishable}, notes={item.human_notes}")

        try:
            prompt = "Score (0-4) [s/b/p/q]: "
            choice = input(prompt).strip().lower()

            if choice in ("q", "quit", "exit"):
                print("\nSaving session and exiting...")
                break

            if choice in ("b", "back"):
                if current_idx > 0:
                    current_idx -= 1
                else:
                    print("Already at the first candidate.")
                continue

            if choice in ("s", "skip"):
                current_idx += 1
                continue

            if choice in ("p", "preview"):
                if video_path and video_path.is_file():
                    print(f"Rendering preview clip {item.start:.1f}s - {item.end:.1f}s...")
                    try:
                        clip_path = preview_clip(video_path, item.start, item.end, open_player=True)
                        print(f"Preview saved to: {clip_path}")
                    except Exception as e:
                        print(f"Preview error: {e}")
                else:
                    print("Video file not specified or not found. Pass --video to enable video preview.")
                continue

            score_input_val = None
            try:
                val = float(choice)
                if 0.0 <= val <= 4.0:
                    score_input_val = int(val) if val.is_integer() else val
            except ValueError:
                pass

            if score_input_val is not None:
                score = score_input_val

                # Prompt for publishable
                curr_pub_str = "y" if item.publishable is True else ("n" if item.publishable is False else "none")
                pub_input = input(f"Publishable as-is? [y/n] (current: {curr_pub_str}): ").strip().lower()
                if pub_input == "y":
                    publishable = True
                elif pub_input == "n":
                    publishable = False
                elif pub_input == "" and item.publishable is not None:
                    publishable = item.publishable
                else:
                    publishable = score >= 3  # reasonable default

                # Prompt for notes
                curr_note_str = f"'{item.human_notes}'" if item.human_notes else "none"
                notes_input = input(f"Notes (optional, Enter to keep {curr_note_str}): ").strip()
                if notes_input:
                    notes = notes_input
                elif item.human_notes:
                    notes = item.human_notes
                else:
                    notes = None

                # Update candidate item
                item.human_score = score
                item.publishable = publishable
                item.human_notes = notes

                eval_doc.update_labeled_count()

                # Atomic save immediately on each answer
                save_json(eval_doc, eval_file_path)
                print(f"Saved: Candidate {item.candidate_id} scored as {score} (publishable={publishable})")

                current_idx += 1
            else:
                print("Invalid input. Enter a score between 0 and 4, 's' to skip, 'b' to go back, 'p' to preview, or 'q' to quit.")

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Progress saved.")
            break

    eval_doc.update_labeled_count()
    save_json(eval_doc, eval_file_path)
    print(f"\nAnnotation saved. Total labeled: {eval_doc.labeled_candidates}/{total}")
    return eval_doc.labeled_candidates
