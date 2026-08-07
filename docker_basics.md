
# Docker Basics — Starting From Absolute Zero

**You have installed Docker Desktop and you have no idea what it is. That is
exactly the right place to start.**

Read this before we touch any project. Twenty minutes. No cloud background,
no DevOps background, no prior tools needed.

By the end you should be able to explain, in your own words, what Docker is and
why anyone bothers with it.

---

## 1. Start with a real problem

Arjun is a fresher at **Meridian Semiconductor Advisory**, a research desk that
tracks supply-chain risk in the chip industry. He has spent three weeks building
a Graph RAG application that answers questions like *"if ASML stops shipping,
who gets hurt?"*

It works beautifully on his laptop.

His manager says: *"Great. Send it to Priya so she can demo it to the client
tomorrow."*

Arjun zips the folder and sends it. Then this happens.

| Priya says                                 | The actual reason                                              |
| ------------------------------------------ | -------------------------------------------------------------- |
| "It says`ModuleNotFoundError: chromadb`" | She never ran`pip install -r requirements.txt`               |
| "Now it says something about Python 3.9"   | Arjun built it on Python 3.11                                  |
| "The PDF loading crashed"                  | She is on Windows, he is on a Mac                              |
| "It can't find the data folder"            | She ran it from inside`scripts/` instead of the project root |
| "Some SSL certificate error?"              | A Mac-specific fix Arjun applied months ago and forgot about   |

Four hours gone. The demo is tomorrow.

And notice: **the code was never the problem.** Not one line of Arjun's code was
wrong. Every single failure was about the *machine* the code was running on.

> ### The sentence that started an industry
>
> *"But it works on my machine."*
>
> Docker exists to make that sentence meaningless.

---

## 2. So what is Docker?

> **In one sentence:** Docker packs your application together with everything
> it needs to run — the right Python, the right libraries, the right folder
> layout, the right settings — into one sealed box that behaves identically on
> any computer.

Arjun does not send Priya his code. He sends her the sealed box. She opens it
and it runs, because everything it needed came inside.

### Two ways to picture it

**The lunchbox.** Sending someone a recipe means hoping they own the right pan,
the right spices, the right stove. Sending them a packed lunchbox means they
just open it and eat. Docker sends the lunchbox.

**The shipping container.** Before shipping containers, moving goods across the
world meant repacking cargo for every truck, ship and train. Then everyone
agreed on one standard steel box. Suddenly it did not matter what was inside —
every port, crane and truck could handle it the same way. Docker is that box for
software. That is also why its logo is a whale carrying containers.

### What actually goes inside the box

```
┌─────────────────────────────────────┐
│  YOUR SEALED BOX                    │
│                                     │
│   Your code                         │
│   Python 3.11 (the exact version)   │
│   Every library, exact versions     │
│   The folder structure              │
│   Settings and start command        │
│                                     │
└─────────────────────────────────────┘
        ↓ runs identically on ↓

   Arjun's Mac    Priya's Windows    A server in Mumbai
```

That is the whole idea. Everything after this is detail.

---

## 3. The one distinction that matters: image vs container

Beginners mix these up constantly, and almost every confusing Docker error
traces back to it. Learn it now and the rest is easy.

> **An image is the recipe. A container is the dish you cooked from it.**

|                    | Image                               | Container                                       |
| ------------------ | ----------------------------------- | ----------------------------------------------- |
| What it is         | A sealed, frozen package            | A running copy of that package                  |
| Does it run?       | No. It just sits there.             | Yes. It is alive.                               |
| How many?          | One                                 | As many as you like, all from the same image    |
| Can you change it? | No, it is fixed                     | Yes, but changes vanish when it stops           |
| Real-world twin    | A recipe, or an app in an app store | Tonight's dinner, or the app open on your phone |

One recipe, many dinners. One image, many containers.

```
     IMAGE  ──────▶  container 1   (running)
   (the box)  ──────▶  container 2   (running)
                ──────▶  container 3   (stopped)
```

**Why "changes vanish" matters.** Start a container, create a file inside it,
stop the container — the file is gone. Containers are disposable by design. This
surprises everyone once. Section 6 explains how to keep things that should
survive.

---

## 4. The Dockerfile — writing the recipe

A **Dockerfile** is a plain text file where you write the steps to build your
image. It is instructions for a very literal assistant. Here is a complete one:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]
```

Seven instructions. Line by line:

| Instruction                 | In plain English                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------- |
| `FROM python:3.11-slim`   | Start from a ready-made box that already has Python 3.11. Never start from nothing. |
| `WORKDIR /app`            | Work inside a folder called`/app`. Every command after this runs from there.      |
| `COPY requirements.txt .` | Copy just the shopping list in, first.                                              |
| `RUN pip install ...`     | Install the libraries. This happens**while building the box**, once.          |
| `COPY . .`                | Now copy the actual code in.                                                        |
| `EXPOSE 8000`             | Note that this app talks on door number 8000.                                       |
| `CMD [...]`               | The command to run**when someone starts a container**.                        |

### Two things worth noticing

**Why is `requirements.txt` copied separately, before the code?**

Docker remembers each step. If nothing about a step changed, it reuses last
time's result instead of redoing it. Installing libraries is slow; editing code
is constant. By copying the shopping list first, Docker only reinstalls
libraries when the list actually changes — so rebuilds after a code edit take
seconds instead of minutes.

This ordering is deliberate in almost every real Dockerfile you will ever read.

**`RUN` and `CMD` sound the same but are not.**

`RUN` happens once, while building the image. `CMD` happens every time a
container starts. Installing goes in `RUN`. Starting your app goes in `CMD`.

### Building and running

```bash
docker build -t finsight .        # read the Dockerfile, make the image
docker run finsight               # start a container from it
```

`-t finsight` names the image. The `.` means "the Dockerfile is in this folder".

---

## 5. Your box is sealed — so how does anything get in or out?

A container is deliberately cut off from your computer. That is the point: it
cannot be affected by your machine, so it behaves the same everywhere.

But a sealed box is useless. Four holes are punched in it deliberately.

### Hole 1 — Ports (letting people reach your app)

Your app runs on port 8000 **inside** the container. Your browser cannot see
inside. So you connect a door on your machine to a door on the container:

```bash
docker run -p 8000:8000 finsight
```

Read it as *"my 8000 → its 8000"*. Now `localhost:8000` in your browser reaches
the app.

You can map different numbers — `-p 9000:8000` means visit `localhost:9000`.
Useful when something already occupies 8000 on your machine.

> **The most common beginner bug.** Inside a container, your app must listen on
> `0.0.0.0`, not `localhost`. `localhost` inside a container means "only things
> inside this box" — which excludes your browser. This is why the Dockerfile
> above says `--host 0.0.0.0`. Leave it out and the app runs perfectly and is
> completely unreachable.

### Hole 2 — Volumes (keeping data that should survive)

Remember: when a container stops, everything inside it is lost.

That is fine for the app. It is a disaster for a database, an uploaded file, or
a vector index that took ten minutes to build.

A **volume** connects a folder on your computer to a folder inside the
container. Anything written there is really written to your machine, so it
outlives the container.

```bash
docker run -v ./data:/app/data finsight
```

Read it as *"my `data` folder → its `/app/data` folder"*.

```
        WITHOUT a volume                    WITH a volume

   container writes data                container writes data
            ↓                                     ↓
      inside the box                     your real data folder
            ↓                                     ↓
   container stops → GONE               container stops → still there
```

### Hole 3 — Environment variables (passing in secrets and settings)

Your app needs an OpenAI key and a database password. Those must **never** be
written inside the Dockerfile, because anyone who gets the image gets your keys.

Instead you hand them in when the container starts:

```bash
docker run --env-file .env finsight
```

Same principle you already know from `aws configure`: the code lives in one
place, the secrets live somewhere else, and they meet only at run time.

### Hole 4 — The registry (sending your box to someone else)

A **registry** is an app store for images. **Docker Hub** is the big public one.

```bash
docker pull nginx        # download someone else's image
docker push myimage      # upload yours
```

This is how Arjun finally gets his app to Priya. He pushes the image; she pulls
it and runs it. No Python install, no pip, no version mismatch.

It is also where `python:3.11-slim` came from in the Dockerfile. You were
already using the registry without knowing it.

---

## 6. What is Docker Desktop, then?

You installed it. Here is what it actually is.

- **The Docker Engine** — the part that does the real work: building images and
  running containers. All the commands above talk to this.
- **A window with buttons** — see your images and containers, start and stop
  them, read their logs, without typing anything.
- **A small hidden Linux machine** — containers are a Linux technology. On Mac
  and Windows, Docker Desktop quietly runs a tiny Linux system to host them.
  You never see it, but it is why Docker Desktop is a bigger install than you
  might expect.

Use the terminal to learn, and the GUI to look around. Everything the buttons do
is also a command, and vice versa.

---

## 7. Is this the same as a virtual machine?

Close, and the difference is the reason Docker took over.

A **virtual machine** is a whole fake computer — its own full operating system,
booting from scratch. Heavy: gigabytes of disk, a minute to start.

A **container** shares your computer's existing operating system and packs only
what your app needs on top. Light: a few hundred megabytes, starts in a second.

|                      | Virtual machine            | Container                       |
| -------------------- | -------------------------- | ------------------------------- |
| Contains             | An entire operating system | Just your app and its libraries |
| Size                 | Several gigabytes          | Tens to hundreds of megabytes   |
| Start time           | Around a minute            | Around a second                 |
| How many on a laptop | Two or three               | Dozens                          |

**When you still want a VM:** when you need a genuinely different operating
system, or the strongest possible isolation between workloads.

---

## 8. Why companies actually use it

Beyond "it works on my machine", four reasons come up constantly:

- **Every environment matches.** The image a developer tested is the identical
  image that runs in testing and in production. No "it passed testing but broke
  live".
- **New joiners start on day one.** No two-day setup document. Clone the repo,
  run one command, working environment.
- **Scaling is copy-paste.** Traffic tripled? Run more containers from the same
  image. This is what Kubernetes automates once you have more containers than a
  human can track.
- **Nothing pollutes your laptop.** Try Postgres, Redis, Neo4j — run them,
  delete the containers, your machine is untouched. No half-uninstalled
  services lurking for months.

---

## 9. When Docker is the wrong choice

Being honest about this matters, because beginners over-apply it.

- **A one-file script you run occasionally.** Wrapping it in Docker is more work
  than the script.
- **Learning Python on your own laptop.** Docker adds a layer between you and
  your code that will confuse you while you are still learning the basics.
- **Desktop applications with a window.** Containers are built for
  command-line and server programs. Graphical apps are painful.
- **You need a different operating system.** That is a virtual machine's job.
- **Heavy GPU work, sometimes.** It is possible, but the setup is fiddly and not
  a beginner's first project.

---

## 10. The alternatives

You will hear these names. Now they will not be intimidating.

| Name                     | What it is                                                                                                      |
| ------------------------ | --------------------------------------------------------------------------------------------------------------- |
| **Podman**         | Docker without a background service running as administrator. Same commands.                                    |
| **containerd**     | The low-level engine underneath Docker. You rarely touch it directly.                                           |
| **Kubernetes**     | Not an alternative — the layer above. It runs hundreds of containers across many machines. Learn Docker first. |
| **Docker Compose** | Also not an alternative. A file that describes several containers so one command starts them all.               |

Everyone follows the same open standard, so an image built with Docker runs
under Podman and vice versa.

---

## 11. The commands you will actually use

Nine of these cover almost everything.

```bash
# BUILDING
docker build -t myapp .              # build an image from the Dockerfile here

# RUNNING
docker run myapp                     # start a container
docker run -p 8000:8000 myapp        # ... and connect port 8000
docker run -it myapp bash            # ... and open a shell inside it, to look around

# LOOKING
docker ps                            # what is running right now
docker ps -a                         # everything, including stopped
docker images                        # what images do I have
docker logs <name>                   # what did this container print

# CLEANING UP
docker stop <name>                   # stop a running container
docker rm <name>                     # delete a stopped container
docker system prune                  # delete all unused containers and images
```

`docker run -it myapp bash` is worth trying early. It drops you *inside* the
container at a command prompt. Run `ls` and you are looking at your app's world
from the inside. Nothing makes containers feel real faster.

---

## 12. Words you will hear

| Word                        | Plain meaning                                                             |
| --------------------------- | ------------------------------------------------------------------------- |
| **Image**             | The sealed, frozen package. The recipe.                                   |
| **Container**         | A running copy of an image. The dish.                                     |
| **Dockerfile**        | The text file listing how to build an image.                              |
| **Build**             | Turning a Dockerfile into an image.                                       |
| **Base image**        | The ready-made image you start from, like`python:3.11-slim`.            |
| **Layer**             | One step of a build. Docker reuses unchanged layers to save time.         |
| **Volume**            | A folder shared between your machine and the container, so data survives. |
| **Port mapping**      | Connecting a door on your machine to a door on the container.             |
| **Registry**          | An app store for images. Docker Hub is the big public one.                |
| **Tag**               | A label on an image, usually a version —`myapp:v2`.                    |
| **Docker Compose**    | A file describing several containers, started together.                   |
| **Kubernetes**        | The system that runs containers at large scale. Comes much later.         |
| **`.dockerignore`** | A list of files to keep OUT of the image — secrets, caches, junk.        |

---

## 13. Before the hands-on session

You should be able to answer these three out loud. If any is shaky, go back to
that section.

1. A friend asks *"what is Docker?"* — what do you say in two sentences?
2. What is the difference between an image and a container?
3. Arjun's container ran an ingestion that took ten minutes. He restarted it and
   the data was gone. What did he forget, and why does it work that way?

---

## What happens next

We take the real Meridian Graph RAG application — FastAPI, Neo4j, ChromaDB,
five agents, the lot — and put it in a box.

You will meet every idea from this document again, but arriving because the
application forces it: a port because the API needs reaching, a volume because
ChromaDB must survive, environment variables because there is an OpenAI key,
and a `.dockerignore` because that key must never end up inside the image.

None of it will be new. It will just be real.

---

*Tarka Upskilling and Engineering Co. · tarkaupskilling.com*
