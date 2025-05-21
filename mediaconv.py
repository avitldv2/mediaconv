#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import logging
import magic
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def setup_logging(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger('mediaconv')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

class MediaConverter:
    SUPPORTED_FORMATS = {
        'audio': ['mp3', 'wav', 'flac', 'ogg', 'm4a'],
        'video': ['mp4', 'avi', 'mkv', 'mov', 'gif', 'webm']
    }
    
    DEFAULT_SETTINGS = {
        'audio': {
            'bitrate': '192k',
            'sample_rate': '44100',
            'channels': '2'
        },
        'video': {
            'bitrate': '192k',
            'sample_rate': '44100',
            'channels': '2',
            'crf': '23',
            'preset': 'medium'
        }
    }
    
    def __init__(self):
        self.logger = setup_logging()
        
        if not self._check_ffmpeg():
            raise EnvironmentError("FFmpeg is not installed. Please install FFmpeg first.")
        if not self._check_magic():
            raise EnvironmentError("python-magic is not installed. Please install python-magic first.")

    def _check_ffmpeg(self) -> bool:
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _check_magic(self) -> bool:
        try:
            magic.Magic(mime=True)
            return True
        except (ImportError, AttributeError):
            return False

    def _detect_media_type(self, file_path: str) -> Tuple[str, str]:
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            
            if file_type.startswith('audio/'):
                return 'audio', file_path.split('.')[-1].lower()
            elif file_type.startswith('video/'):
                return 'video', file_path.split('.')[-1].lower()
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise ValueError(f"Error detecting file type: {str(e)}")

    def _validate_input_file(self, input_file: str) -> None:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.access(input_file, os.R_OK):
            raise PermissionError(f"Cannot read input file: {input_file}")

    def _validate_output_format(self, media_type: str, output_format: str) -> None:
        output_format = output_format.lower()
        
        if media_type == 'video' and output_format in self.SUPPORTED_FORMATS['audio']:
            return
            
        if output_format not in self.SUPPORTED_FORMATS[media_type]:
            raise ValueError(f"Unsupported output format for {media_type}: {output_format}")

    def _generate_output_filename(
        self,
        input_path: Path,
        output_format: str,
        output_name: Optional[str] = None,
        add_suffix: bool = False,
        output_dir: Optional[str] = None
    ) -> Path:
        if output_name:
            output_path = Path(output_name).with_suffix(f'.{output_format}')
        else:
            stem = input_path.stem
            if add_suffix:
                stem = f"{stem}_converted"
            output_path = input_path.with_name(f"{stem}.{output_format}")
        
        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            return output_dir_path / output_path.name
        
        return output_path

    def _build_ffmpeg_command(
        self,
        input_file: str,
        output_file: str,
        settings: Optional[Dict[str, str]] = None,
        image_file: Optional[str] = None,
        overwrite: bool = False
    ) -> List[str]:
        cmd = ['ffmpeg']
        if overwrite:
            cmd.append('-y')
        output_file_str = str(output_file)
        
        if image_file and output_file_str.endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
            is_gif = image_file.lower().endswith('.gif')
            
            if is_gif:
                cmd.extend(['-stream_loop', '-1', '-i', image_file])
            else:
                cmd.extend(['-loop', '1', '-i', image_file])
            
            cmd.extend(['-i', input_file])
            cmd.extend(['-map', '0:v:0', '-map', '1:a:0'])
            cmd.extend(['-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2'])
            
            if is_gif:
                cmd.extend(['-r', '25'])
            
            if output_file_str.endswith('.webm'):
                cmd.extend(['-c:v', 'libvpx-vp9'])
                cmd.extend(['-crf', '30'])
                cmd.extend(['-b:v', '0'])
                cmd.extend(['-c:a', 'libopus'])
            else:
                cmd.extend(['-c:v', 'libx264', '-tune', 'stillimage'])
                cmd.extend(['-c:a', 'aac'])
            
            cmd.extend(['-pix_fmt', 'yuv420p', '-shortest'])
        else:
            cmd.extend(['-i', input_file])
            if output_file_str.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                cmd.append('-vn')
        
        if settings:
            if output_file_str.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.webm')):
                if output_file_str.endswith('.webm'):
                    cmd.extend(['-c:a', 'libopus'])
                elif output_file_str.endswith('.mp3'):
                    cmd.extend(['-c:a', 'libmp3lame'])
                elif output_file_str.endswith('.flac'):
                    cmd.extend(['-c:a', 'flac'])
                elif output_file_str.endswith('.ogg'):
                    cmd.extend(['-c:a', 'libvorbis'])
                elif output_file_str.endswith('.m4a'):
                    cmd.extend(['-c:a', 'aac'])
                
                if 'audio_bitrate' in settings:
                    cmd.extend(['-b:a', settings['audio_bitrate']])
                if 'sample_rate' in settings:
                    cmd.extend(['-ar', settings['sample_rate']])
                if 'channels' in settings:
                    cmd.extend(['-ac', settings['channels']])
                
                if output_file_str.endswith('.mp3') and 'quality' in settings:
                    cmd.extend(['-q:a', settings['quality']])
                elif output_file_str.endswith('.ogg') and 'quality' in settings:
                    cmd.extend(['-q:a', settings['quality']])
                elif output_file_str.endswith('.webm') and 'quality' in settings:
                    cmd.extend(['-vbr', settings['quality']])
            
            elif output_file_str.endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
                if output_file_str.endswith('.webm'):
                    cmd.extend(['-c:v', 'libvpx-vp9'])
                else:
                    cmd.extend(['-c:v', 'libx264'])
                
                if 'video_bitrate' in settings:
                    cmd.extend(['-b:v', settings['video_bitrate']])
                if 'audio_bitrate' in settings:
                    cmd.extend(['-b:a', settings['audio_bitrate']])
                if 'crf' in settings:
                    if output_file_str.endswith('.webm'):
                        crf = min(63, max(0, int(settings['crf'])))
                        cmd.extend(['-crf', str(crf)])
                    else:
                        crf = min(51, max(0, int(settings['crf'])))
                        cmd.extend(['-crf', str(crf)])
                if 'preset' in settings:
                    if output_file_str.endswith('.webm'):
                        cmd.extend(['-cpu-used', settings['preset']])
                    else:
                        cmd.extend(['-preset', settings['preset']])
                
                if output_file_str.endswith('.webm'):
                    if 'tile-columns' in settings:
                        cmd.extend(['-tile-columns', settings['tile-columns']])
                    if 'frame-parallel' in settings:
                        cmd.extend(['-frame-parallel', settings['frame-parallel']])
                else:
                    if 'profile' in settings:
                        cmd.extend(['-profile:v', settings['profile']])
                    if 'level' in settings:
                        cmd.extend(['-level', settings['level']])
            
            elif output_file_str.endswith('.gif'):
                if 'duration' in settings:
                    cmd.extend(['-t', settings['duration']])
                cmd.extend([
                    '-vf', 'fps=15,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                    '-loop', '0'
                ])
        
        cmd.append(output_file_str)
        return cmd

    def convert(
        self,
        input_file: str,
        output_format: str,
        settings: Optional[Dict[str, str]] = None,
        image_file: Optional[str] = None,
        add_suffix: bool = False,
        dry_run: bool = False,
        overwrite: bool = False,
        output_name: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> str:
        try:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            if image_file and not os.path.exists(image_file):
                raise FileNotFoundError(f"Image file not found: {image_file}")

            if output_format == 'gif':
                try:
                    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_entries', 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                    if not result.stdout.strip():
                        raise ValueError(f"Input file '{input_file}' does not contain any video stream")
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Error checking video stream: {e.stderr}")

            if output_format in ['mp3', 'wav', 'flac', 'ogg', 'm4a']:
                try:
                    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                    if not result.stdout.strip():
                        raise ValueError(f"Input file '{input_file}' does not contain any audio stream")
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Error checking audio stream: {e.stderr}")

            output_file = self._generate_output_filename(
                Path(input_file),
                output_format,
                output_name=output_name,
                add_suffix=add_suffix,
                output_dir=output_dir
            )
            
            if os.path.exists(output_file) and not overwrite and not dry_run:
                while True:
                    response = input(f"File {output_file} already exists. Overwrite? [y/N]: ").lower()
                    if response in ['y', 'yes']:
                        overwrite = True
                        break
                    elif response in ['n', 'no', '']:
                        raise FileExistsError(f"Output file already exists: {output_file}")
                    else:
                        print("Please answer 'y' or 'n'")
            
            if overwrite and os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except PermissionError:
                    import time
                    time.sleep(1)
                    try:
                        os.remove(output_file)
                    except PermissionError as e:
                        raise RuntimeError(f"Cannot overwrite file {output_file}: {str(e)}")
            
            cmd = self._build_ffmpeg_command(input_file, output_file, settings, image_file, overwrite)
            
            if dry_run:
                print(f"Would execute: {' '.join(cmd)}")
                return str(output_file)
            
            try:
                duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
                duration = float(subprocess.run(duration_cmd, capture_output=True, text=True, check=True).stdout.strip())
            except (subprocess.CalledProcessError, ValueError):
                duration = None
            
            try:
                if duration is not None:
                    with tqdm(total=100, desc="Converting", unit="%") as pbar:
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            bufsize=1
                        )
                        
                        stderr_output = []
                        def read_stderr():
                            for line in process.stderr:
                                stderr_output.append(line)
                                if 'time=' in line:
                                    try:
                                        time_str = line.split('time=')[1].split()[0]
                                        h, m, s = time_str.split(':')
                                        time_sec = float(h) * 3600 + float(m) * 60 + float(s)
                                        progress = min(100, round((time_sec / duration) * 100))
                                        pbar.update(progress - pbar.n)
                                    except (ValueError, ZeroDivisionError):
                                        continue
                        
                        import threading
                        stderr_thread = threading.Thread(target=read_stderr)
                        stderr_thread.daemon = True
                        stderr_thread.start()
                        
                        process.wait()
                        stderr_thread.join(timeout=1)
                        
                        if process.returncode != 0:
                            error_msg = ''.join(stderr_output)
                            raise RuntimeError(f"Conversion failed: {error_msg}")
                else:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    if result.stderr:
                        print(f"FFmpeg output: {result.stderr}")
                
                return str(output_file)
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else str(e)
                raise RuntimeError(f"Conversion failed: {error_msg}")
        except Exception as e:
            self.logger.error("Error converting %s: %s", input_file, str(e))
            raise

    def convert_batch(
        self,
        input_files: List[str],
        output_format: str,
        output_dir: Optional[str] = None,
        settings: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
        dry_run: bool = False,
        max_workers: int = 4,
        add_suffix: bool = False
    ) -> Dict[str, List[str]]:
        self.logger.info("Starting batch conversion of %d files", len(input_files))
        
        results = {'success': [], 'failed': []}
        
        if dry_run:
            for input_file in input_files:
                try:
                    output_file = self.convert(
                        input_file=input_file,
                        output_format=output_format,
                        settings=settings,
                        add_suffix=add_suffix,
                        dry_run=True,
                        overwrite=overwrite,
                        output_dir=output_dir
                    )
                    results['success'].append(output_file)
                except Exception as e:
                    self.logger.error("Failed to convert %s: %s", input_file, str(e))
                    results['failed'].append(input_file)
            return results
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for input_file in input_files:
                future = executor.submit(
                    self.convert,
                    input_file=input_file,
                    output_format=output_format,
                    settings=settings,
                    add_suffix=add_suffix,
                    overwrite=overwrite,
                    output_dir=output_dir
                )
                futures.append((input_file, future))
            
            with tqdm(total=len(futures), desc="Converting files", unit="file") as pbar:
                for input_file, future in futures:
                    try:
                        output_file = future.result()
                        if output_file:
                            results['success'].append(output_file)
                        else:
                            results['failed'].append(input_file)
                    except Exception as e:
                        self.logger.error("Failed to convert %s: %s", input_file, str(e))
                        results['failed'].append(input_file)
                    finally:
                        pbar.update(1)
        
        success_count = len(results['success'])
        failed_count = len(results['failed'])
        self.logger.info("Batch conversion complete: %d succeeded, %d failed", 
                        success_count, failed_count)
        
        if failed_count > 0:
            self.logger.error("\nFailed conversions:")
            for failed_file in results['failed']:
                self.logger.error("  - %s", failed_file)
        
        return results

def main() -> int:
    parser = argparse.ArgumentParser(description='Convert media files using FFmpeg')
    parser.add_argument('input_files', nargs='+', help='Input file(s) to convert')
    parser.add_argument('--to', required=True, help='Output format (e.g., mp3, wav, mp4)')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--output-name', help='Custom output filename (for single input)')
    
    parser.add_argument('--audio-bitrate', help='Audio bitrate (e.g., 192k)')
    parser.add_argument('--video-bitrate', help='Video bitrate (e.g., 2M)')
    parser.add_argument('--sample-rate', help='Audio sample rate (e.g., 44100)')
    parser.add_argument('--channels', choices=['1', '2'], help='Number of audio channels')
    parser.add_argument('--quality', help='Audio quality (0-9 for MP3, 0-10 for OGG)')
    
    parser.add_argument('--crf', help='Video quality (0-51 for H.264, 0-63 for VP9)')
    parser.add_argument('--preset', choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                       help='FFmpeg preset')
    parser.add_argument('--profile', choices=['baseline', 'main', 'high', 'high10', 'high422', 'high444'],
                       help='H.264 profile')
    parser.add_argument('--level', help='H.264 level (e.g., 4.0, 4.1)')
    
    parser.add_argument('--tile-columns', help='WebM tile columns (0-4)')
    parser.add_argument('--frame-parallel', choices=['0', '1'], help='WebM frame parallel')
    
    parser.add_argument('--duration', help='Duration in seconds for GIF conversion (max 15)')
    
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files (default: False)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show commands without executing')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel conversions')
    parser.add_argument('--add-suffix', action='store_true',
                       help='Add "_converted" suffix to output files')
    parser.add_argument('--test', action='store_true',
                       help='Run environment checks and exit')
    parser.add_argument('--image', help='Image file to use when converting audio to video')

    args = parser.parse_args()

    if args.to == 'gif' and args.duration:
        try:
            duration = float(args.duration)
            if duration <= 0 or duration > 15:
                print("Error: GIF duration must be between 0 and 15 seconds")
                return 1
        except ValueError:
            print("Error: Duration must be a valid number")
            return 1

    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return 1
        if not os.access(input_file, os.R_OK):
            print(f"Error: Cannot read input file: {input_file}")
            return 1

    if args.output_dir:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            if not os.access(args.output_dir, os.W_OK):
                print(f"Error: Cannot write to output directory: {args.output_dir}")
                return 1
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return 1

    if args.output_name and len(args.input_files) > 1:
        print("Warning: --output-name is ignored for batch conversion")

    try:
        converter = MediaConverter()
        
        if args.test:
            print("Environment check passed!")
            return 0

        settings = {}
        if args.audio_bitrate:
            settings['audio_bitrate'] = args.audio_bitrate
        if args.video_bitrate:
            settings['video_bitrate'] = args.video_bitrate
        if args.sample_rate:
            settings['sample_rate'] = args.sample_rate
        if args.channels:
            settings['channels'] = args.channels
        if args.quality:
            settings['quality'] = args.quality
            
        if args.crf:
            settings['crf'] = args.crf
        if args.preset:
            settings['preset'] = args.preset
        if args.profile:
            settings['profile'] = args.profile
        if args.level:
            settings['level'] = args.level
            
        if args.tile_columns:
            settings['tile-columns'] = args.tile_columns
        if args.frame_parallel:
            settings['frame-parallel'] = args.frame_parallel
            
        if args.duration and args.to == 'gif':
            settings['duration'] = args.duration

        if len(args.input_files) == 1:
            try:
                output_file = converter.convert(
                    input_file=args.input_files[0],
                    output_format=args.to,
                    settings=settings,
                    image_file=args.image,
                    add_suffix=args.add_suffix,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                    output_name=args.output_name,
                    output_dir=args.output_dir
                )
                if output_file:
                    print(f"Successfully converted to: {output_file}")
                return 0 if output_file else 1
            except FileExistsError as e:
                print(f"Error: {e}")
                print("Use --overwrite to force overwrite existing files")
                return 1
            except Exception as e:
                print(f"Error: {str(e)}")
                return 1
        else:
            results = converter.convert_batch(
                input_files=args.input_files,
                output_format=args.to,
                output_dir=args.output_dir,
                settings=settings,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                max_workers=args.max_workers,
                add_suffix=args.add_suffix
            )
            
            success_count = len(results['success'])
            failed_count = len(results['failed'])
            print(f"Converted {success_count} out of {len(args.input_files)} files. {failed_count} failed.")
            
            if failed_count > 0:
                print("\nFailed conversions:")
                for failed_file in results['failed']:
                    print(f"  - {failed_file}")
            
            return 0 if success_count == len(args.input_files) else 1

    except EnvironmentError as e:
        print(f"Environment error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 