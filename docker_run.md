
# Running FINSIGHT in Docker — Every Command, Explained

**From an empty terminal to a live API, one command at a time.**

Every step below follows the same shape:

> **What you need first** → **The command** → **What it means** → **What you should see**

Read `DOCKER_BASICS.md` before this. It explains images, containers, ports and
volumes. This file assumes you know those four words.

---

## Contents

| Part                                                    | What happens                          |
| ------------------------------------------------------- | ------------------------------------- |
| [Part 0](#part-0--check-your-setup)                      | Check Docker is actually running      |
| [Part 1](#part-1--get-the-project-and-its-secrets)       | Get the project and its secrets ready |
| [Part 2](#part-2--build-the-image)                       | Build the image                       |
| [Part 3](#part-3--run-the-ingestion)                     | Run the ingestion (build the graph)   |
| [Part 4](#part-4--start-the-api)                         | Start the API and use it              |
| [Part 5](#part-5--look-inside-a-running-container)       | Look inside a running container       |
| [Part 6](#part-6--stop-and-clean-up)                     | Stop and clean up                     |
| [Part 7](#part-7--the-same-thing-with-docker-compose)    | The same thing, with Docker Compose   |
| [Experiment](#experiment--what-happens-without-a-volume) | Watch data disappear without a volume |

---

# Part 0 — Check your setup

## 0.1 Is Docker installed?

**What you need first:** Docker Desktop installed.

```bash
docker --version
```

**What it means:** Asks Docker to state its version. The simplest possible
check that the command exists.

**Expected output:**

```
Docker version 27.3.1, build ce12230
```

Your numbers will differ. Any version is fine.

> **If you see `command not found`** — Docker Desktop is installed but your
> terminal cannot find it. Close the terminal completely and open a new one. If
> it still fails, open Docker Desktop once and let it finish starting.

---

## 0.2 Is Docker actually running?

**What you need first:** the previous command worked.

```bash
docker info
```

**What it means:** Asks the Docker engine — the part that does the real work —
to describe itself. `docker --version` only proves the command exists;
this proves the engine behind it is awake.

**Expected output:** about forty lines of detail, beginning something like:

```
Client:
 Version:    27.3.1
Server:
 Containers: 0
  Running: 0
 Images: 0
 Server Version: 27.3.1
```

`Containers: 0` and `Images: 0` are correct for a fresh install. You have not
built anything yet.

> **If you see `Cannot connect to the Docker daemon`** — Docker Desktop is not
> running. Open the application and wait for the whale icon to stop animating.
> This is the single most common Docker error and it always means the same
> thing: the engine is asleep.

---

# Part 1 — Get the project and its secrets

## 1.1 Clone the repository

```bash
git clone https://github.com/SetuAI/Supplychain-Graph-RAG.git
cd Supplychain-Graph-RAG
```

**What it means:** Downloads the project and moves into its folder.

**Expected output:**

```
Cloning into 'Supplychain-Graph-RAG'...
remote: Enumerating objects: 84, done.
Receiving objects: 100% (84/84), 2.14 MiB | 4.20 MiB/s, done.
```

**Everything from here on must be run from inside this folder.** Confirm with:

```bash
ls
```

You should see `Dockerfile`, `requirements.txt`, `api`, `graph_rag`, `scripts`
and `data`. If you do not see `Dockerfile`, you are in the wrong folder.

---

## 1.2 Create your .env file

**What you need first:** an OpenAI API key, and a Neo4j Aura instance with its
connection details.

```bash
cp .env.example .env
```

**What it means:** Makes your own copy of the settings template. `.env.example`
is a blank form committed to the repository; `.env` is your filled-in copy,
which is never committed.

**Expected output:** nothing. Silence means success.

Now open `.env` in your editor and fill in the real values:

```
OPENAI_API_KEY=sk-your-real-key
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-real-password
```

Leave the rest as they are.

> ### Why this file is not going into the image
>
> An image can be shared, pushed to a registry, and downloaded by anyone. If
> your keys were inside it, they would travel with it.
>
> `.dockerignore` keeps `.env` out of the image. We pass it in separately at
> run time with `--env-file`. Same principle as `aws configure`: code in one
> place, secrets in another, meeting only when the program actually runs.

---

# Part 2 — Build the image

## 2.1 Build

**What you need first:** you are inside the project folder, and `Dockerfile`
is visible when you run `ls`.

```bash
docker build -t finsight:v1 .
```

**What it means:**

| Part               | Meaning                                                                    |
| ------------------ | -------------------------------------------------------------------------- |
| `docker build`   | Read a Dockerfile and produce an image                                     |
| `-t finsight:v1` | Name the result`finsight`, version tag `v1`                            |
| `.`              | The Dockerfile is in this folder — the dot is the folder, not a full stop |

That final dot is easy to miss and the command fails without it.

**Expected output:** a stream of steps, one per Dockerfile instruction:

```
[+] Building 184.2s (11/11) FINISHED
 => [internal] load build definition from Dockerfile           0.0s
 => [internal] load .dockerignore                              0.0s
 => [1/5] FROM docker.io/library/python:3.11-slim              8.4s
 => [2/5] WORKDIR /app                                         0.1s
 => [3/5] COPY requirements.txt .                              0.0s
 => [4/5] RUN pip install --no-cache-dir --upgrade pip && ...  162.3s
 => [5/5] COPY . .                                             0.3s
 => exporting to image                                         5.1s
 => => naming to docker.io/library/finsight:v1
```

**The first build takes several minutes.** Most of it is step 4 installing
LangChain, ChromaDB and the rest. This is normal and it happens once.

> **Now prove the layer caching lesson.** Run the exact same command again:
>
> ```bash
> docker build -t finsight:v1 .
> ```
>
> It finishes in a couple of seconds, with `CACHED` next to most steps. Docker
> checked each layer, found nothing had changed, and reused all of them. This
> is why `requirements.txt` is copied before the code — so editing your code
> never re-triggers that 162-second install.

---

## 2.2 Confirm the image exists

```bash
docker images
```

**What it means:** Lists every image on your machine.

**Expected output:**

```
REPOSITORY   TAG       IMAGE ID       CREATED          SIZE
finsight     v1        a3f8c91e2b47   2 minutes ago    1.24GB
python       3.11-slim 9d2c7e4a1f83   3 weeks ago      151MB
```

Two images, and that is worth understanding. `python:3.11-slim` was downloaded
because your Dockerfile said `FROM` it. `finsight:v1` is yours, built on top.

The size will surprise you. Roughly 150 MB is Python, and the rest is the
libraries this project needs.

---

# Part 3 — Run the ingestion

This is the build step for your **data** — it reads the PDFs, embeds them into
ChromaDB, and writes the knowledge graph into Neo4j Aura. It runs once.

## 3.1 Run ingestion inside a container

**What you need first:** a built image, a filled-in `.env`, and a working
internet connection.

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  finsight:v1 \
  python scripts/run_ingestion.py
```

**What it means, flag by flag:**

| Part                                | Meaning                                                                                                                         |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `docker run`                      | Start a container from an image                                                                                                 |
| `--rm`                            | Delete the container automatically when it finishes. This is a one-off job, not a service — no need to keep the corpse around. |
| `--env-file .env`                 | Read your secrets from`.env` and hand them to the container as environment variables                                          |
| `-v "$(pwd)/data:/app/data"`      | Connect your real`data` folder to `/app/data` inside the container. Read it as **my folder : its folder**.            |
| `finsight:v1`                     | Which image to start                                                                                                            |
| `python scripts/run_ingestion.py` | **Replaces** the Dockerfile's `CMD`. Instead of starting the API, run the ingestion script.                             |

That last line is the point worth pausing on. **One image, two different
jobs.** You did not build a second image for ingestion — you simply told this
container to run a different command.

`$(pwd)` means "the folder I am in right now". On Windows PowerShell use
`${PWD}` instead.

**Expected output:** several minutes of progress, ending with:

```
Connected to Neo4j AuraDB successfully
[PASS] TSMC supplies chips to Apple
[PASS] ASML supplies equipment to TSMC
[PASS] TSMC manufactures for Nvidia
[PASS] TSMC operates in Taiwan
[PASS] Apple depends on TSMC
Neo4j connection closed
```

Each `[PASS]` confirms a relationship exists in the graph. These are the exact
hops the demo questions walk, so five passes means the graph can answer them.

> **If you see `ServiceUnavailable` or a routing timeout** — Neo4j Aura
> connects on port 7687, and many office and college networks block it. Docker
> does not change this; the container uses your machine's network. Test on a
> phone hotspot to confirm, then ask your network team to allow outbound 7687
> to `*.databases.neo4j.io`.

---

## 3.2 Prove the volume worked

**What you need first:** ingestion finished.

```bash
ls data/chroma_db
```

**What it means:** Looks at the folder on **your own machine**, not inside the
container. The container that wrote these files no longer exists — `--rm`
deleted it.

**Expected output:**

```
chroma.sqlite3
f47ac10b-58cc-4372-a567-0e02b2c3d479
```

This is the volume doing its job. A deleted container, and its work survives —
because it was written through to your real folder.

Skip to [the experiment](#experiment--what-happens-without-a-volume) at the end
of this file to see what happens when you leave the volume out. It is the
clearest five minutes in this whole walkthrough.

---

# Part 4 — Start the API

## 4.1 Start it in the background

```bash
docker run -d \
  --name finsight-api \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  finsight:v1
```

**What it means:**

| Part                    | Meaning                                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `-d`                  | Detached. Run in the background and give the terminal back. Without this, the container holds your terminal until you stop it. |
| `--name finsight-api` | Give the container a name you can type later. Without it, Docker invents one like`nostalgic_hopper`.                         |
| `-p 8000:8000`        | Connect port 8000 on your machine to port 8000 in the container.**My port : its port.**                                  |
| `--env-file .env`     | The secrets again — every container needs them                                                                                |
| `-v ...`              | The data folder again, so the API can read what ingestion built                                                                |
| `finsight:v1`         | The image                                                                                                                      |

Notice there is **no command at the end this time**. That means the Dockerfile's
`CMD` runs — the uvicorn line — which is exactly what we want.

**Expected output:** one long line of letters and numbers.

```
7f3a9c2e8b1d4a6f5c9e2b8d1a4f7c3e9b2d5a8f1c4e7b3a6d9f2c5e8b1a4d7f
```

That is the container's full ID. Its appearance means the container started.

---

## 4.2 Check it is running

```bash
docker ps
```

**What it means:** Lists containers that are **currently running**. Add `-a` to
include stopped ones.

**Expected output:**

```
CONTAINER ID   IMAGE         COMMAND                  STATUS         PORTS                    NAMES
7f3a9c2e8b1d   finsight:v1   "uvicorn api.main:ap…"   Up 8 seconds   0.0.0.0:8000->8000/tcp   finsight-api
```

Three things to read here:

- **STATUS `Up`** — it is alive. If it says `Exited (1)`, something crashed; go
  to 4.3 and read the logs.
- **COMMAND** — the uvicorn line from your Dockerfile's `CMD`, confirming the
  default ran.
- **PORTS** — `0.0.0.0:8000->8000/tcp` is your port mapping, working.

---

## 4.3 Read the logs

**What you need first:** a container named `finsight-api`.

```bash
docker logs finsight-api
```

**What it means:** Shows everything the container has printed. Since it is
running in the background, this is your only window into it.

**Expected output:**

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Note `http://0.0.0.0:8000` — that is the `--host 0.0.0.0` flag from the
Dockerfile. Had it said `127.0.0.1`, your browser would never reach it.

To watch the logs live as requests arrive:

```bash
docker logs -f finsight-api
```

`-f` means follow. Press `Ctrl+C` to stop watching — this does **not** stop the
container, only your view of it.

---

## 4.4 Use the API

Open this in your browser:

```
http://localhost:8000/docs
```

**What it means:** FastAPI generates an interactive page listing every endpoint,
where you can send real requests without writing any code.

**Expected output:** a page titled with your API name, listing the `/health`
and `/query` endpoints.

Or test it from the terminal:

```bash
curl http://localhost:8000/health
```

**Expected output:**

```json
{"status":"ok"}
```

> **Pause here.** You are talking to an application that is not installed on
> your computer. There is no Python environment for it, no `pip install` was
> run on your machine, and no libraries were added. It is running inside a
> sealed box, and your browser reaches it through one deliberate hole you
> punched with `-p`.
>
> That is Docker, working.

Now ask it a real question through `/docs` — try *"If ASML stops shipping EUV
machines, which companies face the highest revenue risk?"* — and watch
`docker logs -f finsight-api` in another terminal while it answers.

---

# Part 5 — Look inside a running container

This step is not required. Do it anyway. Nothing makes containers feel real
faster than standing inside one.

```bash
docker exec -it finsight-api bash
```

**What it means:**

| Part             | Meaning                                                         |
| ---------------- | --------------------------------------------------------------- |
| `docker exec`  | Run an extra command inside a container that is already running |
| `-it`          | Interactive terminal — keep it open and let you type           |
| `finsight-api` | Which container                                                 |
| `bash`         | The command to run: a shell                                     |

**Expected output:** your prompt changes.

```
root@7f3a9c2e8b1d:/app#
```

You are now **inside the container**. That hostname is the container ID, and
`/app` is the `WORKDIR` from the Dockerfile.

Look around:

```bash
ls                    # your project files, exactly as copied
ls data/chroma_db     # the ingested data, arriving through the volume
python --version      # Python 3.11.x, regardless of what your laptop has
cat .env              # No such file. The .dockerignore worked.
env | grep OPENAI     # The key IS here - passed in at run time, not baked in
exit                  # leave the container; it keeps running
```

Those last two commands together are the whole secrets lesson in ten seconds.
The file is absent from the image; the value is present in the running
container. That is exactly the separation you want.

---

# Part 6 — Stop and clean up

## 6.1 Stop the container

```bash
docker stop finsight-api
```

**What it means:** Politely asks the container to shut down, giving it ten
seconds to finish what it is doing.

**Expected output:**

```
finsight-api
```

Docker echoes the name of what it stopped. Check with `docker ps` — the list is
now empty.

---

## 6.2 Remove it

**What you need first:** the container is stopped.

```bash
docker rm finsight-api
```

**What it means:** Deletes the stopped container. Without this, the name stays
taken and starting a new one with `--name finsight-api` fails.

**Expected output:**

```
finsight-api
```

The **image** is untouched. You can start a fresh container from it any time.
And your `data/chroma_db` is untouched too, because it was never inside the
container.

---

## 6.3 Clean up everything unused

```bash
docker system prune
```

**What it means:** Deletes all stopped containers, unused networks, and dangling
images. It asks for confirmation first, and it never touches running containers
or images currently in use.

**Expected output:**

```
WARNING! This will remove:
  - all stopped containers
  - all networks not used by at least one container
  - all dangling images
  - all build cache
Are you sure you want to continue? [y/N] y

Total reclaimed space: 2.147GB
```

Worth running every few weeks. Docker accumulates old layers quietly, and it
adds up to gigabytes faster than you expect.

---

# Part 7 — The same thing, with Docker Compose

## What Compose actually is

Look at the command from 4.1 again. Six lines, four flags, easy to mistype, and
impossible to remember next week.

`docker-compose.yml` is that command, written down.

**Nothing runs inside the file. There is no code in it.** It is a settings
sheet. You run commands *against* it, and Compose reads it to work out what
flags you meant.

Every line in it maps to a flag you have already used:

| Line in`docker-compose.yml`    | The flag it replaces           |
| -------------------------------- | ------------------------------ |
| `build: .`                     | `docker build .`             |
| `image: finsight:v1`           | `-t finsight:v1`             |
| `container_name: finsight-api` | `--name finsight-api`        |
| `ports: - "8000:8000"`         | `-p 8000:8000`               |
| `env_file: - .env`             | `--env-file .env`            |
| `volumes: - ./data:/app/data`  | `-v "$(pwd)/data:/app/data"` |
| `restart: unless-stopped`      | `--restart unless-stopped`   |

The file is already in the repository root. Read through it once — it is
commented line by line.

> **Why bother, for one container?** Honestly, for a single container it is a
> convenience rather than a necessity. Compose earns its keep when a project
> needs several containers at once — an app, a database, a cache — started
> together. This project needs only one, because Neo4j lives in the cloud on
> Aura. So treat it as the tidy version of a command you already understand.

---

## 7.1 Check the file is valid before running anything

**What you need first:** `docker-compose.yml` in the project root.

```bash
docker compose config
```

**What it means:** Reads the file, checks it for mistakes, and prints back what
Compose actually understood — with your `.env` values filled in and relative
paths expanded. It does not start anything.

**Expected output:**

```yaml
name: supplychain-graph-rag
services:
  api:
    build:
      context: /Users/you/Supplychain-Graph-RAG
      dockerfile: Dockerfile
    container_name: finsight-api
    environment:
      NEO4J_URI: neo4j+s://xxxxxxxx.databases.neo4j.io
      OPENAI_API_KEY: sk-...
    image: finsight:v1
    ports:
      - mode: ingress
        target: 8000
        published: "8000"
    restart: unless-stopped
    volumes:
      - type: bind
        source: /Users/you/Supplychain-Graph-RAG/data
        target: /app/data
```

Run this first whenever something behaves unexpectedly. YAML is fussy about
indentation, and this tells you immediately whether Compose read what you
thought you wrote.

> **Note your secrets are printed in full here.** Do not run this while
> screen-sharing or recording.

---

## 7.2 Build the image through Compose

```bash
docker compose build
```

**What it means:** Finds the `build: .` line, reads the Dockerfile, and builds
the image — the same work as `docker build -t finsight:v1 .`, with the name
taken from the file.

**Expected output:**

```
[+] Building 3.2s (11/11) FINISHED
 => [internal] load build definition from Dockerfile           0.0s
 => CACHED [2/5] WORKDIR /app                                  0.0s
 => CACHED [3/5] COPY requirements.txt .                       0.0s
 => CACHED [4/5] RUN pip install --no-cache-dir --upgrade ...  0.0s
 => CACHED [5/5] COPY . .                                      0.0s
 => => naming to docker.io/library/finsight:v1
```

If you already built in Part 2, this finishes in seconds and every step says
`CACHED`. Compose and `docker build` share the same cache — they are the same
engine underneath.

This step is optional. `docker compose up` builds automatically if no image
exists yet.

---

## 7.3 Run the ingestion through Compose

**What you need first:** a filled-in `.env`, and internet access.

```bash
docker compose run --rm api python scripts/run_ingestion.py
```

**What it means:**

| Part                                | Meaning                                                          |
| ----------------------------------- | ---------------------------------------------------------------- |
| `docker compose run`              | Start a**one-off** container using this service's settings |
| `--rm`                            | Delete it when the job finishes                                  |
| `api`                             | Which service's settings to borrow — the name from the file     |
| `python scripts/run_ingestion.py` | The command to run, replacing the Dockerfile's`CMD`            |

This is the Compose version of the command from Part 3.1. You still get the
`.env` and the volume automatically, because they are written in the file.

**Use `run`, not `up`, for one-off jobs.** `up` is for services that keep
running; `run` is for a task that finishes and exits.

**Expected output:** several minutes of progress, ending with:

```
Connected to Neo4j AuraDB successfully
[PASS] TSMC supplies chips to Apple
[PASS] ASML supplies equipment to TSMC
[PASS] TSMC manufactures for Nvidia
[PASS] TSMC operates in Taiwan
[PASS] Apple depends on TSMC
Neo4j connection closed
```

> **Why does `run` not map the port?** By default `docker compose run` skips
> the `ports` section, so a one-off job never collides with a service already
> using port 8000. Sensible, and occasionally surprising.

---

## 7.4 Start the API

```bash
docker compose up -d
```

**What it means:** Reads the file, builds the image if needed, and starts every
service in it. `-d` detaches so you get your terminal back.

**Expected output:**

```
[+] Running 1/1
 ✔ Container finsight-api  Started                              0.4s
```

That single command replaced all six lines from Part 4.1.

To rebuild and start in one go after changing your code:

```bash
docker compose up -d --build
```

Without `--build`, Compose reuses the existing image and your code changes are
not picked up. This trips people up constantly.

---

## 7.5 Check what is running

```bash
docker compose ps
```

**What it means:** Like `docker ps`, but only shows containers belonging to
this project. Useful once you have several projects on one machine.

**Expected output:**

```
NAME           IMAGE         COMMAND                  SERVICE   STATUS         PORTS
finsight-api   finsight:v1   "uvicorn api.main:ap…"   api       Up 12 seconds  0.0.0.0:8000->8000/tcp
```

Note the extra `SERVICE` column — that is the name from your file.

---

## 7.6 Read the logs

```bash
docker compose logs -f api
```

**What it means:** Shows what the service has printed. `-f` follows it live.
Drop the `api` to see logs from every service at once — more useful when a
project has several.

**Expected output:**

```
finsight-api  | INFO:     Started server process [1]
finsight-api  | INFO:     Waiting for application startup.
finsight-api  | INFO:     Application startup complete.
finsight-api  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Compose prefixes each line with the container name. With one service that
looks redundant; with four it is the only thing keeping the output readable.

`Ctrl+C` stops watching, not the container.

---

## 7.7 Go inside the running container

```bash
docker compose exec api bash
```

**What it means:** The Compose version of `docker exec -it`. You name the
**service**, not the container, and `-it` is assumed.

**Expected output:**

```
root@7f3a9c2e8b1d:/app#
```

Same as Part 5 — look around with `ls`, `cat .env` (absent), `env | grep OPENAI`
(present), then `exit`.

---

## 7.8 Stop and remove

```bash
docker compose down
```

**What it means:** Stops every container in the project and removes them, plus
the network Compose created. The image stays. Your `data` folder stays.

**Expected output:**

```
[+] Running 2/2
 ✔ Container finsight-api        Removed                        10.3s
 ✔ Network supplychain-graph-rag_default  Removed                0.1s
```

This is `docker stop` and `docker rm` in one command.

To stop without removing, so you can start again quickly:

```bash
docker compose stop
docker compose start
```

> ### One flag to be careful with
>
> `docker compose down -v` also deletes volumes.
>
> In **this** project that is harmless — our volume is a bind mount to your
> real `./data` folder, and Compose will not delete that. But in projects using
> named volumes, `-v` destroys the database. Know what your volumes are before
> you type it.

---

## Compose quick reference

```bash
docker compose config                # check the file is valid
docker compose build                 # build the image
docker compose up -d                 # start, in the background
docker compose up -d --build         # rebuild first, then start
docker compose ps                    # what is running
docker compose logs -f api           # watch the output
docker compose exec api bash         # go inside
docker compose run --rm api python scripts/run_ingestion.py   # one-off job
docker compose stop                  # stop, keep the containers
docker compose down                  # stop and remove
```

> **`docker compose` or `docker-compose`?** Older tutorials use the hyphen —
> that was a separate program. Modern Docker Desktop builds it in, and the
> command is `docker compose` with a space. Both may work on your machine; use
> the space.

---

# Experiment — What happens without a volume

Do this once. It teaches volumes better than any explanation.

**Step 1 — Run ingestion with no `-v` flag:**

```bash
docker run --name temp-test --env-file .env finsight:v1 \
  python scripts/run_ingestion.py
```

Wait for the `[PASS]` lines.

**Step 2 — Check the data is there, inside that container:**

```bash
docker exec temp-test ls /app/data/chroma_db
```

Nothing shows, because the container has already exited. So start a fresh one:

```bash
docker run --rm --env-file .env finsight:v1 ls /app/data/chroma_db
```

**Expected output:**

```
ls: cannot access '/app/data/chroma_db': No such file or directory
```

**The ten minutes of ingestion are gone.** It happened inside a container, the
container ended, and everything written inside ended with it.

**Step 3 — Now do it with the volume:**

```bash
docker run --rm --env-file .env -v "$(pwd)/data:/app/data" finsight:v1 \
  python scripts/run_ingestion.py
```

Then, on your own machine:

```bash
ls data/chroma_db
```

**Expected output:**

```
chroma.sqlite3
f47ac10b-58cc-4372-a567-0e02b2c3d479
```

Same work, same container lifecycle, completely different outcome. The volume
routed every write through to your real folder instead of into the disposable
box.

**Clean up the leftover:**

```bash
docker rm temp-test
```

---

# Quick reference

```bash
# BUILD
docker build -t finsight:v1 .

# INGEST (once)
docker run --rm --env-file .env -v "$(pwd)/data:/app/data" \
  finsight:v1 python scripts/run_ingestion.py

# SERVE
docker run -d --name finsight-api -p 8000:8000 --env-file .env \
  -v "$(pwd)/data:/app/data" finsight:v1

# INSPECT
docker ps                          # what is running
docker logs -f finsight-api        # what is it saying
docker exec -it finsight-api bash  # go inside

# STOP
docker stop finsight-api
docker rm finsight-api

# OR, WITH COMPOSE
docker compose up -d
docker compose down
```

---

# When something breaks

| What you see                            | What it means                                      | What to do                                                         |
| --------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------ |
| `Cannot connect to the Docker daemon` | Docker Desktop is not running                      | Open the app, wait for the whale to settle                         |
| `failed to read dockerfile`           | You are in the wrong folder, or the dot is missing | `ls` to check for `Dockerfile`; check the trailing `.`       |
| `error: command 'gcc' failed`         | A library needs a compiler the slim image lacks    | Uncomment the`build-essential` block in the Dockerfile           |
| `port is already allocated`           | Something else is on 8000                          | Use`-p 8001:8000` and visit `localhost:8001`                   |
| `name is already in use`              | An old container still holds the name              | `docker rm finsight-api`                                         |
| Container exits immediately             | The app crashed on startup                         | `docker logs finsight-api` — the reason is always there         |
| Browser shows nothing, logs look fine   | Missing`-p`, or the app bound to `127.0.0.1`   | Check`docker ps` shows a PORTS mapping; check `--host 0.0.0.0` |
| `ServiceUnavailable` from Neo4j       | Port 7687 blocked by your network                  | Test on a hotspot; ask IT to allow outbound 7687                   |
| `OPENAI_API_KEY` errors               | `--env-file .env` was left off                   | Add it — every container needs it                                 |

---

*Tarka Upskilling and Engineering Co. · tarkaupskilling.com*
