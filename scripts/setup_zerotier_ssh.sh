#!/usr/bin/env bash
# ==============================================================================
# One-Click ZeroTier + Planet + Static Node ID + SSH Setup for Cloud Instances
# Network: cb9fa0c138690227 (brea) | Subnet: 172.25.0.0/24
# ==============================================================================
set -e

# Default root password if not provided as argument
PASSWORD="${1:-wavfry123456}"
NETWORK_ID="${2:-cb9fa0c138690227}"

echo "=========================================================="
echo " [ZeroTier & SSH Setup] Starting automated setup..."
echo " Target Network ID : ${NETWORK_ID}"
echo " SSH Root Password : ${PASSWORD}"
echo "=========================================================="

# 0. Bypass systemd/dbus service auto-start timeout in Docker containers
echo -e '#!/bin/sh\nexit 101' > /usr/sbin/policy-rc.d
chmod +x /usr/sbin/policy-rc.d

# 1. Install OpenSSH Server & prerequisites
echo "[1/6] Installing OpenSSH Server and network utilities..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    openssh-server curl gpg sudo >/dev/null

# 2. Configure SSH service for Root Password Login
echo "[2/6] Configuring SSH daemon..."
mkdir -p /var/run/sshd
echo "root:${PASSWORD}" | chpasswd
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
# Ensure SSH host keys are generated in fresh containers
ssh-keygen -A 2>/dev/null || true
# Directly launch sshd (bypasses systemctl/dbus wait in containers)
/usr/sbin/sshd 2>/dev/null || service ssh restart 2>/dev/null || true
echo "      SSH service is running."

# 3. Check / Create TUN Device for Container
echo "[3/6] Checking virtual TUN device..."
mkdir -p /dev/net
if [ ! -c /dev/net/tun ]; then
    mknod /dev/net/tun c 10 200 2>/dev/null || true
    chmod 666 /dev/net/tun 2>/dev/null || true
fi
if [ -c /dev/net/tun ]; then
    echo "      /dev/net/tun is available."
else
    echo "      [WARNING] /dev/net/tun is missing or restricted. ZeroTier might require container host network/tun support."
fi

# 4. Install ZeroTier-One
echo "[4/6] Installing ZeroTier-One..."
if ! command -v zerotier-one >/dev/null 2>&1; then
    echo "      Running ZeroTier installer..."
    (curl -sSL --connect-timeout 8 https://install.zerotier.com | bash) 2>/dev/null || {
        echo "      Direct installer failed or timed out, setting up official deb repository..."
        DISTRO=$(. /etc/os-release && echo "${VERSION_CODENAME:-bookworm}")
        mkdir -p /usr/share/keyrings
        (curl -sSL --connect-timeout 8 'https://ghfast.top/https://raw.githubusercontent.com/zerotier/ZeroTierOne/main/doc/contact%40zerotier.com.gpg' | gpg --dearmor -o /usr/share/keyrings/zerotier.gpg) 2>/dev/null || true
        echo "deb [signed-by=/usr/share/keyrings/zerotier.gpg] http://download.zerotier.com/debian/${DISTRO} ${DISTRO} main" > /etc/apt/sources.list.d/zerotier.list
        apt-get update -qq && apt-get install -y -qq zerotier-one
    }
fi

# Clean up policy override now that all packages are installed
rm -f /usr/sbin/policy-rc.d 2>/dev/null || true

# Stop service to deploy planet & identities
pkill -9 zerotier-one 2>/dev/null || true
sleep 1
mkdir -p /var/lib/zerotier-one /data/zerotier

# 5. Deploy Custom Planet & Restore Persistent Identity for Static IP
echo "[5/6] Deploying custom planet and node identity..."

# Embed Base64 planet from C:\ProgramData\ZeroTier\One\planet
cat << 'EOF' | base64 -d > /var/lib/zerotier-one/planet
AQAAAAAI6skKAAABbOPiOVU70eKh3745Fll2gqlJ4LmbU8s7y9Pa73893QT56dmoFxhM0yh/P5P2ph+nWN/FrC1bzjw7xOaayT0r30eH57YlnVlttmxsnLNTy9F0ebmmB0aDTKtqFdKMcZrygfcDv4GbhXvBuLvbjiPEVj900UTJuos5cXVWkzRlmPrnLwlZDAbNnr7q39L4hHBwxk6q7vCx8E718i//VROgQDHr//K1AcufoME4ADAqZodfE0WNTbjkHjdSyVV0Hv7ju3cgYMAJR23tgpFDcm99FgzDm2rgoFojiQ9S1WqYz6pnAdgmXGi7qkEJxPYAAgQmaR+1JwoGKgm6xWNrLaUAAAAABIwAOycK
EOF
cp -f /var/lib/zerotier-one/planet /data/zerotier/planet 2>/dev/null || true

# Key step for Fixed IP: If /data/zerotier/identity.secret exists, reuse it!
if [ -f "/data/zerotier/identity.secret" ]; then
    echo "      Found persistent ZeroTier identity in /data/zerotier/! Restoring for identical Node ID..."
    cp -f /data/zerotier/identity.secret /var/lib/zerotier-one/identity.secret
    cp -f /data/zerotier/identity.public /var/lib/zerotier-one/identity.public
    chmod 600 /var/lib/zerotier-one/identity.secret
else
    echo "      No previous identity found in /data/zerotier/. Generating fresh identity..."
    if command -v zerotier-idtool >/dev/null 2>&1; then
        (cd /var/lib/zerotier-one && zerotier-idtool generate identity.secret identity.public)
        cp -f /var/lib/zerotier-one/identity.* /data/zerotier/ 2>/dev/null || true
        chmod 600 /var/lib/zerotier-one/identity.secret
        echo "      Identity generated and saved to /data/zerotier/ for future reuse."
    fi
fi

# Start ZeroTier daemon (daemon mode directly, avoiding systemctl wait)
zerotier-one -d 2>/dev/null || service zerotier-one start 2>/dev/null || true
sleep 2

# Backup identity if not backed up yet
if [ ! -f "/data/zerotier/identity.secret" ] && [ -f "/var/lib/zerotier-one/identity.secret" ]; then
    cp -f /var/lib/zerotier-one/identity.* /data/zerotier/ 2>/dev/null || true
    echo "      Node identity safely persisted to /data/zerotier/ for permanent IP fixation."
fi

# 6. Join ZeroTier Network
echo "[6/6] Joining ZeroTier Network ${NETWORK_ID}..."
zerotier-cli join "${NETWORK_ID}"

echo ""
echo "=========================================================="
echo " ZeroTier Node Information:"
NODE_ID=$(zerotier-cli info | awk '{print $3}')
echo " Node ID        : ${NODE_ID}"

echo " Waiting for IP assignment from controller (up to 12s)..."
ZT_IP=""
for i in $(seq 1 12); do
    sleep 1
    ZT_IP=$(zerotier-cli listnetworks 2>/dev/null | grep "${NETWORK_ID}" | awk '{print $NF}' | tr ',' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n 1 || true)
    if [ -n "${ZT_IP}" ]; then
        break
    fi
done

if [ -n "${ZT_IP}" ]; then
    CLEAN_IP="${ZT_IP%/*}"
    echo " Assigned ZT IP : ${CLEAN_IP}"
    echo "=========================================================="
    echo " 🎉 SUCCESS! You can now connect via SSH directly from Windows:"
    echo "    ssh root@${CLEAN_IP}"
    echo "    Password: ${PASSWORD}"
    echo "=========================================================="
else
    echo " Status         : Network joined! Waiting for controller authorization / IP allocation."
    echo " If your network requires authorization, please authorize Node ID [${NODE_ID}] in your controller panel."
    echo " Check status anytime with: zerotier-cli listnetworks"
    echo "=========================================================="
fi
