# MediaConv

A simple but powerful Python CLI tool for converting media files using FFmpeg. Supports batch conversion, parallel processing, and configurable settings.

## Features

- Convert between various audio and video formats
- Batch conversion with parallel processing
- Configurable quality settings
- Support for audio-to-video conversion with static images or GIFs

## Installation

### Quick Install

The easiest way to install MediaConv is using the provided installation script:

```bash
# Make the script executable
chmod +x install.sh

# Run the installation script
sudo ./install.sh
```

This will:
1. Install required Python packages
2. Copy the script to `/usr/local/bin`
3. Make it executable
4. Check for FFmpeg installation

After installation, you can use `mediaconv` from anywhere in your system.

### Manual Installation

1. Ensure you have Python 3.6+ installed
2. Install FFmpeg:
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Linux (Debian): `sudo apt-get install ffmpeg`
   - Linux (Arch): `sudo pacman -S ffmpeg`
   - FreeBSD: `sudo pkg install ffmpeg`
   - macOS: `brew install ffmpeg`
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `mediaconv.py` to a directory in your PATH or run it directly

## Usage

### Basic Usage

```bash
mediaconv input.wav --to mp3
```

### Audio-to-Video Conversion

```bash
mediaconv input.mp3 --to mp4 --image cover.jpg
```

### Batch Conversion

```bash
mediaconv *.wav --to mp3 --output-dir converted
```

### Advanced Settings

#### Audio Settings
- `--audio-bitrate`: Audio bitrate (e.g., 192k)
- `--sample-rate`: Sample rate (e.g., 44100)
- `--channels`: Number of channels (1 or 2)
- `--quality`: Codec-specific quality setting
  - MP3: 0-9 (lower is better)
  - OGG: -1 to 10 (higher is better)
  - WebM/Opus: 0-10 (higher is better)

#### Video Settings
- `--video-bitrate`: Video bitrate (e.g., 2M)
- `--crf`: Quality factor
  - H.264: 0-51 (lower is better)
  - VP9: 0-63 (lower is better)
- `--preset`: Encoding preset
  - H.264: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
  - VP9: 0-5 (higher is faster, lower is better quality)
- `--profile`: H.264 profile (baseline, main, high)
- `--level`: H.264 level (e.g., 4.1)
- `--tile-columns`: VP9 tile columns (1-4)
- `--frame-parallel`: VP9 frame parallel mode (0 or 1)

#### GIF Settings
- `--duration`: Duration in seconds (max 15)

### Other Options

- `--output-dir`: Specify output directory
- `--output-name`: Custom output filename (single file only)
- `--overwrite`: Overwrite existing files
- `--dry-run`: Show commands without executing
- `--max-workers`: Maximum parallel conversions (default: 4)
- `--add-suffix`: Add "_converted" suffix to output files
- `--test`: Run environment checks

## Examples

### Convert WAV to MP3 with high quality
```bash
mediaconv input.wav --to mp3 --bitrate 320k --quality 0
```

### Convert to WebM with VP9
```bash
mediaconv input.mp4 --to webm --crf 30 --tile-columns 2 --frame-parallel 1
```

### Convert to H.264 with specific profile
```bash
mediaconv input.mp4 --to mp4 --crf 23 --profile high --level 4.1
```

### Convert audio to video with static image
```bash
mediaconv input.mp3 --to mp4 --image cover.jpg --crf 23
```

### Audio Formats
- MP3
- WAV
- FLAC
- OGG
- M4A

### Video Formats
- MP4
- AVI
- MKV
- MOV
- WEBM
- GIF

## License

This project uses the MIT license. Read [LICENSE](/LICENSE.md) for more info
