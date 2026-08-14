#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y fail2ban
sudo cp "$(dirname "$0")/jail.local" /etc/fail2ban/jail.local
sudo systemctl enable --now fail2ban
sudo systemctl restart fail2ban
sudo fail2ban-client ping
sudo fail2ban-client status
sudo fail2ban-client status sshd
echo "OK: fail2ban активен для sshd"
