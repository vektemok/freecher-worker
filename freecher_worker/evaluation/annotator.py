"""Interactive terminal-based annotation interface for blind human evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Set
from freecher_worker.utils.json_io import save_json
from .audio_preview import AudioPreviewer, AudioPreviewError
from .models import BlindEvaluationDocument, BlindEvaluationItem

#: Structured dimensions asked after the overall score, in order.
#: (attribute, prompt, low anchor, high anchor)
DIMENSION_PROMPTS = (
    ("hook_score", "Hook", "no pull", "irresistible"),
    ("standalone_score", "Standalone", "incomprehensible", "fully self-contained"),
    ("payoff_score", "Payoff", "none", "complete payoff"),
    ("value_score", "Value (emotional/surprising/informative)", "flat", "striking"),
    ("context_dependency", "Context dependency", "none needed", "strongly dependent"),
)


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


def prompt_dimension(label: str, low: str, high: str, current: Optional[float]) -> Optional[float]:
    """Ask for one 0-4 dimension; Enter keeps whatever is already there.

    Returns None when nothing has been recorded, which is different from 0:
    unrecorded means the rater did not judge it, not that it scored badly.
    """
    suffix = f" (current: {current:g})" if current is not None else ""
    while True:
        raw = input(f"  {label} [0={low} .. 4={high}]{suffix}: ").strip()
        if raw == "":
            return current
        try:
            value = float(raw)
        except ValueError:
            print("  Enter a number between 0 and 4, or press Enter to leave it unset.")
            continue
        if 0.0 <= value <= 4.0:
            return int(value) if value.is_integer() else value
        print("  Out of range. Enter a number between 0 and 4.")


def prompt_boolean(label: str, current: Optional[bool]) -> Optional[bool]:
    """Ask a yes/no question; Enter keeps the existing answer."""
    current_label = "y" if current is True else ("n" if current is False else "unset")
    while True:
        raw = input(f"  {label} [y/n] (current: {current_label}): ").strip().lower()
        if raw == "":
            return current
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Enter y or n, or press Enter to keep the current answer.")


def run_terminal_annotator(
    eval_doc: BlindEvaluationDocument,
    eval_file_path: Path,
    video_path: Optional[Path] = None,
    re_label_all: bool = False,
    previewer: Optional[AudioPreviewer] = None,
    # Off by default so this stays exactly the loop it was; the CLI turns it
    # on, since the structured dimensions are what pass one now collects.
    capture_dimensions: bool = False,
) -> int:
    """Interactive CLI loop for blind evaluation of candidates.

    Blind throughout: the candidate order is whatever the shuffled document
    carries, and no scorer name, model score or rank is ever shown or read.

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
    print("Scale (overall, and every dimension):")
    print("  0: Unusable (cut mid-sentence, no context, noise)")
    print("  1: Weak (dull, rambling, low interest)")
    print("  2: Acceptable / borderline (coherent, mediocre hook or delivery)")
    print("  3: Good (clear hook, interesting content, strong candidate)")
    print("  4: Excellent (viral hook, punchline/payoff, standalone value)")
    if capture_dimensions:
        print("Then: hook, standalone, payoff, value, context dependency,")
        print("      and whether the window starts or ends badly.")
        print("      Enter alone on a dimension leaves it unrecorded.")
    print("Commands:")
    if previewer is not None:
        print("  [0-4] Score | [s] Skip | [b] Back | [a] Play audio | [x] Stop audio")
        print("  [p] Preview clip | [q] Quit & Save")
    else:
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

            if choice in ("a", "audio"):
                if previewer is None:
                    print("Audio preview is not enabled. Pass --source-id to fetch the audio artifact.")
                else:
                    try:
                        previewer.play(item.start, item.end)
                        print(f"Playing {format_timestamp(item.start)} -> {format_timestamp(item.end)} "
                              f"({item.duration:.1f}s). Press [x] to stop.")
                    except AudioPreviewError as exc:
                        print(f"Audio preview error: {exc}")
                continue

            if choice in ("x", "stop"):
                if previewer is not None:
                    previewer.stop()
                    print("Audio stopped.")
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

                # Update candidate item. human_score stays exactly what the
                # rater said; the dimensions are recorded beside it and never
                # folded back into it.
                item.human_score = score
                item.publishable = publishable
                item.human_notes = notes

                if capture_dimensions:
                    for attribute, label, low, high in DIMENSION_PROMPTS:
                        setattr(
                            item,
                            attribute,
                            prompt_dimension(label, low, high, getattr(item, attribute)),
                        )
                    item.bad_start = prompt_boolean("Bad start (opens mid-thought)?", item.bad_start)
                    item.bad_end = prompt_boolean("Bad end (cuts off the thought)?", item.bad_end)

                eval_doc.update_labeled_count()
                eval_doc.update_dimension_count()

                # Atomic save immediately on each answer
                save_json(eval_doc, eval_file_path)
                print(f"Saved: Candidate {item.candidate_id} scored as {score} (publishable={publishable})")

                if previewer is not None:
                    previewer.stop()
                current_idx += 1
            else:
                print(
                    "Invalid input. Enter a score between 0 and 4, 's' to skip, 'b' to go back, "
                    "'a' to play audio, 'x' to stop it, 'p' to preview, or 'q' to quit."
                )

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Progress saved.")
            break

    if previewer is not None:
        previewer.stop()
    eval_doc.update_labeled_count()
    eval_doc.update_dimension_count()
    save_json(eval_doc, eval_file_path)
    print(f"\nAnnotation saved. Total labeled: {eval_doc.labeled_candidates}/{total}")
    return eval_doc.labeled_candidates


def collect_candidate_ids(source: str) -> Set[str]:
    """Read a candidate-id selection from a comma-separated list or a JSON file.

    A pass-2 subset is normally chosen from something the evaluation tooling
    already produced -- a disagreement report, or the union of scorers' top-K --
    so a path to a JSON array of ids is accepted alongside an inline list.
    """
    path = Path(source)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(entry, str) for entry in payload):
            raise ValueError(
                f"{path} must contain a JSON array of candidate id strings"
            )
        return {entry.strip() for entry in payload if entry.strip()}
    return {part.strip() for part in source.split(",") if part.strip()}


def select_backfill_items(
    items: List[BlindEvaluationItem],
    candidate_ids: Optional[Set[str]] = None,
    include_complete: bool = False,
) -> List[BlindEvaluationItem]:
    """Choose which candidates a dimension backfill pass should revisit.

    Only candidates that already carry a canonical label are eligible: a
    backfill adds detail to a judgement that has been made, it does not make
    one. Without an explicit id set, the pass visits exactly those still
    missing their dimensions, so re-running it converges rather than looping
    over work already done.
    """
    eligible = [item for item in items if item.human_score is not None]
    if candidate_ids is not None:
        eligible = [item for item in eligible if item.candidate_id in candidate_ids]
    if not include_complete:
        eligible = [item for item in eligible if not item.has_dimensions]
    return eligible


def run_dimension_backfill(
    eval_doc: BlindEvaluationDocument,
    eval_file_path: Path,
    candidate_ids: Optional[Set[str]] = None,
    previewer: Optional[AudioPreviewer] = None,
    include_complete: bool = False,
    show_canonical_label: bool = False,
) -> int:
    """Second-pass loop that records only the structured dimensions.

    Deliberately a separate loop from `run_terminal_annotator` rather than a
    branch inside it: this one has no code path that can assign human_score,
    publishable or human_notes, so a pass-one label cannot be damaged by a
    mistyped key no matter what is entered here.

    Blind in the same way as pass one -- no scorer name, model score or rank
    exists in this document at all -- and blind to the rater's own pass-one
    answer as well: seeing "I called this a 4" pulls every dimension towards
    agreeing with it. The canonical values stay in the document untouched;
    `show_canonical_label` only puts them back on screen, for review.

    Returns the number of candidates carrying dimensions when the pass ends.
    """
    targets = select_backfill_items(eval_doc.items, candidate_ids, include_complete)
    total = len(targets)

    if total == 0:
        print("No candidates need dimension labeling.")
        if candidate_ids:
            print("Every selected candidate already has dimensions, or has no canonical label yet.")
        else:
            print("Run pass one first, or pass --redo-dimensions to revisit completed ones.")
        return eval_doc.update_dimension_count()

    print("\n" + "=" * 70)
    print("DIMENSION BACKFILL (pass 2)")
    print(f"Candidate Set: {eval_doc.candidate_set_id}")
    print(f"Candidates to label: {total}")
    print("Scale for every dimension: 0 (lowest) .. 4 (highest)")
    print("human_score, publishable and notes are NOT touched in this pass.")
    if not show_canonical_label:
        print("Your pass-1 answers are hidden so they do not anchor these ones.")
    print("Commands:")
    if previewer is not None:
        print("  [Enter] Label dimensions | [a] Play audio | [x] Stop audio")
        print("  [s] Skip | [b] Back | [q] Quit & Save")
    else:
        print("  [Enter] Label dimensions | [s] Skip | [b] Back | [q] Quit & Save")
    print("=" * 70 + "\n")

    current_idx = 0
    while 0 <= current_idx < total:
        item = targets[current_idx]
        done = eval_doc.update_dimension_count()

        print(f"\n--- [{current_idx + 1}/{total}] (Dimensions recorded: {done}/{len(eval_doc.items)}) ---")
        print(f"Candidate ID: {item.candidate_id}")
        print(f"Timing:       {format_timestamp(item.start)} -> {format_timestamp(item.end)} ({item.duration:.1f}s)")
        print(f"Text:         \"{item.text}\"")
        if show_canonical_label:
            print(f"Your pass-1 label: score={item.human_score}, publishable={item.publishable}")
            if item.human_notes:
                print(f"Your pass-1 notes: {item.human_notes}")

        try:
            choice = input("[Enter] to label, [a/x] audio, [s] skip, [b] back, [q] quit: ").strip().lower()

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

            if choice in ("a", "audio"):
                if previewer is None:
                    print("Audio preview is not enabled. Pass --source-id to fetch the audio artifact.")
                else:
                    try:
                        previewer.play(item.start, item.end)
                        print(f"Playing {format_timestamp(item.start)} -> {format_timestamp(item.end)} "
                              f"({item.duration:.1f}s). Press [x] to stop.")
                    except AudioPreviewError as exc:
                        print(f"Audio preview error: {exc}")
                continue

            if choice in ("x", "stop"):
                if previewer is not None:
                    previewer.stop()
                    print("Audio stopped.")
                continue

            if choice != "":
                print("Unrecognized command. Press Enter to label this candidate's dimensions.")
                continue

            for attribute, label, low, high in DIMENSION_PROMPTS:
                setattr(item, attribute, prompt_dimension(label, low, high, getattr(item, attribute)))
            item.bad_start = prompt_boolean("Bad start (opens mid-thought)?", item.bad_start)
            item.bad_end = prompt_boolean("Bad end (cuts off the thought)?", item.bad_end)

            eval_doc.update_dimension_count()
            # Saved per candidate, so an interrupted pass keeps its work.
            save_json(eval_doc, eval_file_path)
            print(f"Saved dimensions for {item.candidate_id}.")

            if previewer is not None:
                previewer.stop()
            current_idx += 1

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Progress saved.")
            break

    if previewer is not None:
        previewer.stop()
    eval_doc.update_labeled_count()
    eval_doc.update_dimension_count()
    save_json(eval_doc, eval_file_path)
    print(
        f"\nDimension backfill saved. "
        f"{eval_doc.dimension_labeled_candidates}/{len(eval_doc.items)} candidates carry dimensions."
    )
    return eval_doc.dimension_labeled_candidates
