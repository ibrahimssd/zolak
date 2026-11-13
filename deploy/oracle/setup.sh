#!/usr/bin/env bash
# Set up (or update) the Zolna backend on an Oracle Cloud Always Free A1 VM (Ubuntu 22.04, aarch64).
# Idempotent: safe to run again after `git pull` or to change settings.
#
# Usage on the VM:   bash setup.sh ~/zolna-secrets.env
# The secrets file (chmod 600) defines:
#   DUCKDNS_SUBDOMAIN=zolna        DUCKDNS_TOKEN=...        HF_TOKEN=...   (fine-grained token)
# Optional: MODEL_FILE=... (default zolna-allam-v1-Q4_K_M.gguf; base model: ALLaM-7B-Instruct-preview-Q4_K_M.gguf), REPO=... (git URL)
set -euo pipefail

SECRETS="${1:-$HOME/zolna-secrets.env}"
# shellcheck disable=SC1090
source "$SECRETS"
: "${DUCKDNS_SUBDOMAIN:?missing}" "${DUCKDNS_TOKEN:?missing}" "${HF_TOKEN:?missing}"
DOMAIN="${DUCKDNS_SUBDOMAIN}.duckdns.org"
REPO="${REPO:-https://github.com/ibrahimssd/zolna.git}"
MODEL_FILE="${MODEL_FILE:-zolna-allam-v1-Q4_K_M.gguf}"
DIR="$HOME/zolna"

echo "== 1/6 Docker"
if ! command -v docker > /dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi

echo "== 2/6 Firewall (Oracle Ubuntu images only allow SSH in iptables)"
for port in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT 2> /dev/null \
    || sudo iptables -I INPUT 1 -p tcp --dport "$port" -m state --state NEW -j ACCEPT
done
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q iptables-persistent > /dev/null
sudo netfilter-persistent save > /dev/null

echo "== 3/6 Swap (safety margin for 12 GB RAM)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile > /dev/null && sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
fi

echo "== 4/6 Code"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull -q --ff-only; else git clone -q "$REPO" "$DIR"; fi

echo "== 5/6 Configuration + DuckDNS"
cd "$DIR/deploy/oracle"
( umask 077
  cat > .env << EOF
ZOLNA_DOMAIN=$DOMAIN
HF_TOKEN=$HF_TOKEN
MODEL_REPO=ibrahimssd11/zolna-models
MODEL_FILE=$MODEL_FILE
DATA_REPO=ibrahimssd11/zolna-data
FEEDBACK_REPO=ibrahimssd11/zolna-feedback
EOF
)
UPDATE_URL="https://www.duckdns.org/update?domains=${DUCKDNS_SUBDOMAIN}&token=${DUCKDNS_TOKEN}&ip="
echo "DuckDNS update: $(curl -fsS "$UPDATE_URL")"
( crontab -l 2> /dev/null | grep -v 'duckdns.org/update' || true
  echo "*/5 * * * * curl -fsS '$UPDATE_URL' > /dev/null 2>&1" ) | crontab -

echo "== 6/6 Build and start (first build compiles llama.cpp: ~10-15 min on 2 cores)"
sudo docker compose up -d --build
sudo docker compose ps
echo "Done. Follow startup with: sudo docker compose -f $DIR/deploy/oracle/docker-compose.yml logs -f zolna"
echo "Backend URL: https://$DOMAIN/"
