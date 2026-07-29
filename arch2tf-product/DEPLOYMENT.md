# Deploying to EC2 for peer review

This stands up the whole app (frontend + FastAPI backend + Redis, plus
`terraform`/`tflint`/`checkov` inside the backend container) on a single EC2
instance via Docker Compose, fronted by nginx on port 80, so you can share
one link with peers.

**Before you start — the tradeoff you already accepted:** "Apply to
Sandbox" runs a real `terraform apply` using whatever AWS credentials are
active on the instance itself, not the visitor's own account. Anyone with
the link can create (and destroy) real, billable AWS resources in your
account. This runbook still sets up an IAM **instance role** rather than
long-lived access keys (so nothing leaks if the box or an image is ever
shared), and step 3 below sets up a billing alarm as a minimal safety net —
worth keeping even though you've decided to leave the feature open.

## Architecture

```
peer's browser
      │  http://<ec2-public-ip>/
      ▼
  nginx (:80)  ──/api/*──▶  FastAPI backend (:8000)  ──▶  Redis (job state)
      │                            │
  static frontend              terraform / tflint / checkov
  (arch2tf-product/frontend)   (real subprocess calls, incl. real
                                 `terraform apply` against your AWS account)
```

Same-origin by design (nginx serves both the frontend and proxies `/api/`)
so there's no CORS configuration to get right — this works identically at
`http://<public-ip>/`, a real domain, or `localhost` in dev.

## 1. Launch the EC2 instance

Console → EC2 → Launch instance:

- **AMI**: Ubuntu Server 22.04 LTS
- **Instance type**: `t3.small` is enough to try this out; `t3.medium` if
  multiple peers will be testing concurrently (image parsing + terraform
  subprocess calls are the heavy parts)
- **Key pair**: create/select one — you'll need it for SSH
- **Security group**: inbound rules
  - SSH (22) — source: **My IP** only, not `0.0.0.0/0`
  - HTTP (80) — source: `0.0.0.0/0` (this is the link you'll share)
  - Leave 443/8000/6379 closed — nginx is the only public entry point;
    Redis and the backend are only reachable inside the Docker network
- **Storage**: bump the root volume to 20+ GiB (container images, terraform
  provider plugin cache, and uploaded diagrams all add up faster than the
  default 8 GiB)

### IAM instance role (instead of access keys)

IAM → Roles → Create role → AWS service → EC2 → attach a policy scoped to
whatever your sandbox account should actually allow (start narrow — you can
loosen it later; a full `AdministratorAccess` role on a box with an open
public link is a bad combination). Attach the role to the instance either
at launch ("Advanced details" → IAM instance profile) or afterward via
Actions → Security → Modify IAM role.

With the role attached, `apply_runner.py` needs **zero configuration** to
find real AWS credentials — it inherits the container's environment
unchanged, and both `boto3` and `terraform`'s AWS provider automatically
pull temporary credentials from the instance metadata service (IMDS). Leave
the `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` lines in `.env` commented
out — see step 4.

## 2. Fix Docker's IMDS hop limit (important, easy to miss)

EC2's instance metadata service defaults to a hop limit of 1, which is
enough for a process running directly on the host but **not** enough for a
process inside a Docker container (Docker's bridge network adds a hop).
Without this, the `api` container's `terraform apply`/`boto3` calls will
silently fail to find credentials even though the instance role is
attached correctly. Run once, from your own machine (needs the AWS CLI and
credentials with `ec2:ModifyInstanceMetadataOptions`) or from the instance
itself after attaching a role temporarily:

```bash
aws ec2 modify-instance-metadata-options \
  --instance-id <your-instance-id> \
  --http-put-response-hop-limit 2 \
  --http-tokens required
```

## 3. (Recommended) Set a billing alarm

Billing → Budgets → Create budget → cost budget, e.g. $20/month, with an
alert email — a two-line safety net given the link will be open to whoever
you send it to.

## 4. SSH in and install Docker

```bash
ssh -i /path/to/your-key.pem ubuntu@<ec2-public-ip>

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# run docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

## 5. Get the code onto the instance

This repo isn't in git yet, so the simplest path is `rsync` straight from
your machine (run this from your **local** machine, not the EC2 box):

```bash
rsync -avz --exclude 'venv' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'node_modules' \
  -e "ssh -i /path/to/your-key.pem" \
  /Users/priyankamohan/work/thesis/arch2terraform \
  /Users/priyankamohan/work/thesis/arch2tf-product \
  /Users/priyankamohan/work/thesis/docker-compose.yml \
  /Users/priyankamohan/work/thesis/.gitignore \
  ubuntu@<ec2-public-ip>:~/thesis/
```

(If you'd rather use git: push `arch2terraform` and `arch2tf-product` to
two repos, `git clone` both into `~/thesis/` on the instance, plus
`docker-compose.yml`/`.gitignore` from the parent — same end layout either
way, since the Dockerfile's build context assumes both repos as siblings.)

## 6. Configure and start

```bash
ssh -i /path/to/your-key.pem ubuntu@<ec2-public-ip>
cd ~/thesis

cp arch2tf-product/backend/.env.example arch2tf-product/backend/.env
nano arch2tf-product/backend/.env   # review — defaults are fine for a first pass;
                                     # leave the AWS key lines commented out
                                     # since the instance role covers that

docker compose up -d --build
docker compose ps        # all three services should show "healthy"/"running"
docker compose logs -f api   # watch startup, Ctrl-C to stop tailing
```

First build takes a few minutes (downloading terraform/tflint binaries,
installing Python deps, pulling the OpenCV/checkov dependency tree).

## 7. Verify, then share the link

```bash
curl -s http://localhost/api/v1/health
```

should return 200 with a small JSON status blob — that one request proves
nginx is up, correctly proxying `/api/` through to the `api` container, and
the backend itself is responding.

Open `http://<ec2-public-ip>/` in a browser — you should see the same UI
you've been testing locally. That URL is what you share with peers.

## Day-to-day operations

```bash
# tail logs
docker compose logs -f api

# restart just the backend after a code change
rsync -avz ... ubuntu@<ec2-public-ip>:~/thesis/arch2tf-product   # re-sync
docker compose up -d --build api

# stop everything (e.g. overnight, to avoid idle EC2 cost)
docker compose down          # keeps volumes (uploads, job state) — safe to restart later
docker compose down -v       # also wipes volumes — only if you want a clean slate

# check what's actually applied right now in the sandbox AWS account
# (per-job — there's no single global view; check each job's Apply Status
# in the UI, or grep for ApplyStatus.APPLIED across job state in Redis)
docker compose exec redis redis-cli keys 'job:*'
```

## Optional next steps

- **A real domain + HTTPS**: point a domain's A record at the EC2 public
  IP, then run `certbot` (Let's Encrypt) against nginx for a free TLS cert
  — worth doing once this moves past a quick review link, since plain HTTP
  will trip browser warnings for anyone testing from a network that flags
  mixed content.
- **Auto-start on reboot**: `docker compose` services with
  `restart: unless-stopped` (already set) come back up automatically after
  a Docker/instance restart — no extra systemd unit needed.
- **S3 storage backend** instead of the local Docker volume, if you want
  uploaded diagrams/generated ZIPs to survive even a full instance
  replacement — `STORAGE_BACKEND=s3` + `S3_BUCKET` in `.env` (see
  `app/core/config.py`).
