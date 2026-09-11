# Deploying Freecher

Two long-running processes, one host, systemd.

    freecher-api      accepts POST /jobs and writes a job record. Never renders.
    freecher-worker   claims queued jobs and runs them to DONE.

They share nothing but the job directory, so either can be restarted at any
time without disturbing the other. A worker killed mid-render loses that
render's progress and nothing else: every stage re-checks its own artifact on
the next attempt and skips the ones already done.

## Where each stage runs

| Stage        | Host   | Why |
|--------------|--------|-----|
| INGESTING    | Oracle | Network-bound. The VM pulls at ~166 MB/s; the developer Mac uploads at ~0.3 MB/s. |
| TRANSCRIBING | Kaggle T4 (long) / Oracle (short) | faster-whisper wants a GPU. See below. |
| DISCOVERING  | Oracle | Text and cheap audio statistics. Seconds. |
| RANKING      | Oracle | Produced by the same call as discovery. |
| RENDERING    | Oracle | CPU ffmpeg. The slowest stage by far — measure before promising throughput. |
| UPLOADING    | Oracle | Bandwidth again. |

## Install

From a checkout on the target host:

    sudo ./deploy/install.sh

Idempotent. It installs packages, creates the `freecher` service user,
builds `/opt/freecher/venv`, installs the units and the `freecher-run` helper,
and then runs `preflight`. **If preflight fails, nothing is enabled or
started** — a host that cannot burn subtitles or reach R2 must not accept jobs.

First run writes `/etc/freecher/freecher.env` from the template. Fill in the R2
credentials, then re-run. Layout:

    /opt/freecher/app        code (rsynced from the checkout, no .env)
    /opt/freecher/venv       virtualenv, editable install
    /etc/freecher/           env file, root-only (0750 / 0640)
    /var/lib/freecher/jobs   job records — the durable state
    /var/lib/freecher/runs   per-source artifacts and rendered clips

## Operating

    sudo systemctl status freecher-api freecher-worker
    journalctl -u freecher-worker -f          # SKIP/RUN stage gates live here
    curl -s localhost:8000/health | jq        # 200 ready, 503 degraded
    sudo freecher-run preflight               # the same checks, from a shell

`freecher-run` runs any CLI subcommand as the service user with the service
environment, through `systemd-run` — so the R2 secret is never expanded into a
command line where `ps` would show it.

    sudo freecher-run run-url 'https://...' --top-n 3
    sudo freecher-run transcribe-queue --list

## The GPU transcription handoff

The Oracle VM has no CUDA and two ARM cores. `FREECHER_TRANSCRIBE_BACKEND=auto`
routes accordingly:

* audio at or under `FREECHER_CPU_TRANSCRIPTION_MAX_SECONDS` (default 900) runs
  here on CPU;
* anything longer is handed to the GPU host rather than quietly starting a
  multi-hour CPU decode.

The handoff is R2 itself. The worker writes
`processing/{id}/transcribe_request.json`, sets the job to
`AWAITING_TRANSCRIPT`, and stops. On the GPU host (the existing Kaggle T4
notebook):

    freecher-worker transcribe-queue --once

which drains pending requests through the same `transcribe_from_r2` the
`transcribe` command uses, writing `processing/{id}/transcript.json`. The
worker's idle loop notices the transcript, requeues the job, and every earlier
stage skips.

Kaggle sessions cannot be started programmatically from here, so that notebook
run is manual. That is the only manual step in the pipeline, and a parked job
says exactly what it is waiting for instead of failing or hanging.

Set `FREECHER_TRANSCRIBE_BACKEND=local` plus `FREECHER_ALLOW_CPU_TRANSCRIPTION=true`
to accept CPU transcription of any length — it works, it is just slow.

## Disk lifecycle

`runs/<source_id>/` is where a job's working files land, and nothing in the
pipeline used to delete any of them. Artifacts fall into three classes:

* **durable/remote** — the published clip and the transcript/candidates/
  highlights/manifest objects in R2, plus the job records. Never deleted locally
  by cleanup, and the job store is not touched at all.
* **regenerable local** — the downloaded `source.mp4`, the local copy of a
  published clip, and the small intermediates (`subtitles/`, `words/`,
  `crop_paths/`).
* **shared cache** — model weights under `~/.cache`. Expensive, not per-job,
  never a cleanup target.

A local file is only removed when **both** guards pass: its source belongs to no
job in a non-terminal state, and its remote replacement answers a live HEAD with
a matching size (and sha256 where one was recorded).

```bash
sudo freecher-run cleanup --dry-run --older-than-hours 0   # see the plan
sudo freecher-run cleanup --older-than-hours 24            # routine
sudo freecher-run cleanup --max-disk-usage-gb 10           # oldest-first to a cap
sudo freecher-run cleanup --job-id <id>                    # one job's source
sudo freecher-run cleanup --aggressive                     # also the intermediates
```

Every decision is logged as `DELETE`, `KEEP ... reason=`, or
`SKIP ... active_job`, and `--dry-run` logs exactly what a real run would do.

Nothing runs automatically on startup. Set `FREECHER_CLEANUP_AFTER_DONE=true` to
reclaim a job's regenerable files as soon as its clips are verified in R2.

## Disk guards

Ingest and render check free space first and refuse to start below
`FREECHER_MIN_FREE_DISK_GB` or `FREECHER_MIN_FREE_DISK_PERCENT`, failing the job
with an actionable resource error instead of producing a truncated file. Where
the source reports a size, that is used as an additional check -- never as a
substitute for the real reading. `GET /health` reports the same numbers and goes
degraded on the same thresholds, so a host that would refuse work says so before
anyone submits any.

## Ingest egress

A site can refuse this host's *address* rather than the request. That failure
looks like an ordinary ingest error but has a different cause and a different
remedy, so it is reported as `SourceBlockedError`, excluded from automatic
retry, and carries the setting that fixes it.

Measured on the Oracle host:

| Platform | Direct from Oracle |
|---|---|
| Twitch | accepted |
| RuTube | accepted |
| YouTube | **refused** — "Sign in to confirm you're not a bot" |

YouTube's refusal is IP reputation on the datacenter range. Confirmed not to be
a missing-JavaScript problem: it persists with deno 2.9.6 installed as the EJS
runtime and across the `tv`, `android_vr`, `web_safari`, `tv_simply` and `mweb`
player clients.

```bash
sudo freecher-run egress-check                        # what this host can reach
sudo freecher-run egress-check --proxy socks5://h:1080  # validate a new egress
sudo freecher-run egress-check --url '<some url>'     # one specific source
```

Set `FREECHER_INGEST_PROXY` to route yt-dlp -- both the metadata probe and the
media transfer, deliberately through the same egress -- via an address the site
accepts. Restart the worker afterwards, and re-run `egress-check` to confirm.

## Environment

Every variable is documented in `freecher.env.example`. The ones a deployment
must get right:

| Variable | Why it matters |
|---|---|
| `FREECHER_R2_*` | Ingest, transcripts, discovery artifacts and published clips all live in R2. |
| `FREECHER_FFMPEG_PATH` | Must be a build with libass. Ubuntu 24.04's `ffmpeg` package qualifies. Without it every render fails loudly at the subtitle stage. |
| `FREECHER_JOBS_DIR` | Must match `--jobs-dir` in both units, or the API and the worker use different queues. |
| `FREECHER_INGEST_MAX_HEIGHT` | A 3-hour 1080p60 VOD will fill a 45 GB boot volume. 720 is a safe default. |

## Upgrading

    git pull && sudo ./deploy/install.sh

The venv install is editable, so the code is picked up by restarting the
processes, which `install.sh` does explicitly -- after re-running preflight, so a
bad configuration stops the upgrade before it replaces a working process.

## Scope

One API and one worker on one host. The job store's lock is in-process: correct
for that topology, not a distributed lock. Two workers against the same
directory can claim the same job.
