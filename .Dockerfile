# ==============================================================================
#  FINSIGHT Graph RAG - Dockerfile
# ==============================================================================
#
#  WHAT THIS FILE IS
#
#  A Dockerfile is a recipe. It lists the steps needed to build a sealed box
#  (an "image") containing this application and everything it needs to run.
#
#  Docker reads this file top to bottom and performs each instruction in order.
#  Think of it as instructions for a very literal assistant who has never seen
#  your project and will do exactly what you say, nothing more.
#
#  WHAT WE ARE PACKING
#
#    - Python 3.11
#    - Every library in requirements.txt (LangChain, ChromaDB, FastAPI, ...)
#    - All the application code
#    - The command that starts the API
#
#  WHAT WE ARE DELIBERATELY NOT PACKING
#
#    - The .env file, because it holds the OpenAI key and the Neo4j password.
#      Secrets never go inside an image. They are handed in at run time.
#    - The Neo4j database itself. This project uses Neo4j Aura, which lives in
#      the cloud. The container talks to it over the internet, exactly as your
#      laptop does today.
#    - The ChromaDB data. That is created when you run ingestion, and it is
#      kept on your own machine using a volume so it survives restarts.
#
#  HOW TO USE IT
#
#      docker build -t finsight:v1 .
#
#  Full step-by-step instructions are in docker_run.md.
#
# ==============================================================================


# ------------------------------------------------------------------------------
# STEP 1 - Choose what to start from
# ------------------------------------------------------------------------------
# Every image is built on top of another image. You almost never start from
# nothing - you start from something close to what you need.
#
# "python:3.11-slim" is an official image, prepared by the Python team, that
# already contains a working Python 3.11 on a minimal Linux system.
#
# Why 3.11 exactly? Because that is what this project was built and tested on.
# Pinning the version is the whole point. If we wrote just "python", we would
# get whatever the newest version happens to be on the day someone builds this,
# and we would be back to "it worked last month" problems.
#
# Why "slim"? There are several flavours of the Python image:
#     python:3.11           about 1 GB   - includes many tools we do not need
#     python:3.11-slim      about 150 MB - Python plus the bare essentials
#     python:3.11-alpine    about 50 MB  - smallest, but many Python libraries
#                                          fail to install on it
#
# Slim is the sensible middle: small, and everything still works.
FROM python:3.11-slim


# ------------------------------------------------------------------------------
# STEP 2 - Decide which folder we work in
# ------------------------------------------------------------------------------
# This creates a folder called /app inside the container and moves into it.
# Every instruction after this line runs from inside /app.
#
# This quietly solves a real bug from this project's README: "always run from
# the project root, or the relative paths in .env break". Inside the container
# there is no choice about where you run from - you are always in /app, always
# at the project root. The mistake becomes impossible to make.
#
# The name /app is a convention, not a rule. Almost every Python Dockerfile you
# will read uses it.
WORKDIR /app


# ------------------------------------------------------------------------------
# STEP 3 - Copy in the shopping list, and only the shopping list
# ------------------------------------------------------------------------------
# COPY takes files from your computer and puts them inside the image.
#
#     COPY requirements.txt .
#          ^^^^^^^^^^^^^^^^ ^
#          from your machine |
#                            into the current folder inside the image (/app)
#
# THE OBVIOUS QUESTION: why copy this one file now, instead of copying
# everything at once in step 5?
#
# Because Docker builds in layers, and it remembers them.
#
# Each instruction creates a layer. When you rebuild, Docker checks each layer
# in turn: "did anything this step depends on change?" If not, it reuses last
# time's result instantly instead of doing the work again.
#
# Installing these libraries takes several minutes. Editing your code takes
# seconds and happens constantly.
#
# By copying requirements.txt on its own, the slow install step below only
# depends on that one file. Change your code and rebuild, and Docker skips the
# install entirely - the rebuild takes seconds.
#
# If we copied everything first, every single code edit would trigger a full
# reinstall of LangChain, ChromaDB and the rest. This one reordering is the
# difference between a 5-second rebuild and a 5-minute one.
COPY requirements.txt .


# ------------------------------------------------------------------------------
# STEP 4 - Install the libraries
# ------------------------------------------------------------------------------
# RUN executes a command WHILE THE IMAGE IS BEING BUILT. It happens once, on
# your machine, and the result is baked into the image. It does not happen
# again when someone starts a container.
#
# This is the difference between RUN and CMD (step 8):
#     RUN  = do this while building the box       (once)
#     CMD  = do this when the box is opened       (every time)
#
# Two flags worth understanding:
#
#   --no-cache-dir
#       pip normally keeps a copy of every downloaded package in case you
#       install it again. Inside an image that copy is dead weight we will
#       never use, and it can add hundreds of megabytes. This turns it off.
#
#   --upgrade pip
#       some libraries fail to install on an old pip. One second here saves a
#       confusing error later.
#
# The two commands are joined with && so they become ONE layer rather than two.
# Fewer layers means a smaller image.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# IF THE BUILD FAILS AT THIS STEP:
# Some Python libraries are not shipped ready-made and must be compiled during
# installation, which needs a C compiler. The slim image does not include one.
# If you see an error mentioning "gcc", "error: command 'gcc' failed" or
# "building wheel ... failed", uncomment the block below and rebuild.
#
# It must go BEFORE the pip install lines. It adds roughly 300 MB to the image,
# which is why it is not switched on by default.
#
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends build-essential && \
#     rm -rf /var/lib/apt/lists/*
#
# (The last line deletes the package lists afterwards. They are only needed
#  during installation, and leaving them in wastes space in the image.)


# ------------------------------------------------------------------------------
# STEP 5 - Copy the application code in
# ------------------------------------------------------------------------------
# Now, and only now, we copy the rest of the project.
#
#     COPY . .
#          ^ ^
#          | the current folder inside the image (/app)
#          everything in the folder you ran "docker build" from
#
# THIS LOOKS DANGEROUS, AND IT WOULD BE - except for one file.
#
# "everything" would include your .env file, with your OpenAI key and Neo4j
# password sitting inside the image for anyone who obtains it. It would also
# include __pycache__ folders, your local ChromaDB data, and the .git history.
#
# The .dockerignore file next to this Dockerfile is what prevents that. Docker
# reads it before copying and skips everything listed in it.
#
# .dockerignore does for images exactly what .gitignore does for repositories.
# If you learn one habit from this project, make it this one: write the
# .dockerignore before you write the COPY line.
COPY . .


# ------------------------------------------------------------------------------
# STEP 6 - Write down which port this app uses
# ------------------------------------------------------------------------------
# The FastAPI application listens on port 8000.
#
# Be clear about what EXPOSE does: almost nothing. It does not open the port,
# and it does not connect anything. It is documentation - a note inside the
# image saying "this app talks on 8000", which tools and humans can read.
#
# The port is actually connected when you START a container, with -p:
#
#     docker run -p 8000:8000 finsight:v1
#
# Many beginners add EXPOSE, skip -p, and wonder why the browser shows nothing.
EXPOSE 8000


# ------------------------------------------------------------------------------
# STEP 7 - Tell Python not to hide its output
# ------------------------------------------------------------------------------
# ENV sets an environment variable inside the image.
#
# By default Python collects its print output in a buffer and writes it out in
# batches. On a normal terminal you never notice.
#
# Inside a container it is a real problem: "docker logs" shows nothing while
# your ingestion runs, and you cannot tell whether it is working or frozen.
# Setting this to 1 makes Python print everything immediately.
#
# This is the same idea as sys.stdout.flush() if you have met that before.
ENV PYTHONUNBUFFERED=1


# ------------------------------------------------------------------------------
# STEP 8 - The command that runs when a container starts
# ------------------------------------------------------------------------------
# CMD is the default command. Unlike RUN, this does not happen during the
# build - it happens every time someone starts a container from this image.
#
# Reading the parts:
#
#   uvicorn            the web server that runs FastAPI applications
#   api.main:app       where to find the app - the "app" object inside
#                      api/main.py. This matches the uvicorn command in the
#                      project README.
#   --host 0.0.0.0     THE MOST IMPORTANT FLAG IN THIS FILE. See below.
#   --port 8000        the port to listen on, matching EXPOSE above
#
# WHY --host 0.0.0.0 MATTERS SO MUCH
#
# By default uvicorn listens on 127.0.0.1, which means "only accept connections
# from this same machine". Inside a container, "this same machine" means inside
# the box - which excludes your browser.
#
# The result is the single most common Docker beginner bug: the container runs
# perfectly, the logs look healthy, and the browser shows nothing at all.
#
# 0.0.0.0 means "accept connections from anywhere", which lets your -p port
# mapping actually reach the app.
#
# Note there is no --reload here, unlike the README's local command. --reload
# watches your files for changes and restarts, which is useful while coding and
# pointless in a sealed container.
#
# THIS IS A DEFAULT, NOT A RULE. Anything you type after the image name in
# docker run replaces it. That is how one image runs two different jobs:
#
#     docker run finsight:v1                                  -> starts the API
#     docker run finsight:v1 python scripts/run_ingestion.py   -> builds the graph
#     docker run -it finsight:v1 bash                          -> opens a shell inside
#
# The square brackets are the recommended form. It passes the command straight
# to the system rather than through a shell, which avoids some odd behaviour
# when stopping containers.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]