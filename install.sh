#!/bin/bash

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "Installing required Python packages..."
pip3 install -r requirements.txt

echo "Installing mediaconv to /usr/local/bin..."
cp mediaconv.py /usr/local/bin/mediaconv
chmod +x /usr/local/bin/mediaconv

if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is not installed. Please install FFmpeg:"
    echo "  - Debian/Ubuntu: sudo apt-get install ffmpeg"
    echo "  - Arch Linux: sudo pacman -S ffmpeg"
    echo "  - FreeBSD: sudo pkg install ffmpeg"
    echo "  - macOS: brew install ffmpeg"
fi

echo "Installation complete!"