"""
Gateway Parser Wrapper

Wrapper for parsing gateway message files using GWMReader executable.
Adapted from gateway_parser.py for use in webapp backend.
"""

import gc
import json
import os
import subprocess
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

# Batch size limit to prevent memory issues when processing many files
MAX_FILES_PER_BATCH = 50


class GatewayParserWrapper:
    """
    Utility class for parsing gateway message files using GWMReader executable.
    
    This class handles the execution of the GWMReader.exe parser to process
    gateway message data files and return structured JSON data.
    """

    def __init__(
        self,
        site_name: str,
        file_paths: Union[str, List[str]],
        parser_exe_path: Optional[str] = None,
        temp_base_dir: Optional[str] = None
    ):
        """
        Initialize the Gateway Parser Wrapper.

        Args:
            site_name: Site name for the gateway data
            file_paths: Path to file or list of file paths to process
            parser_exe_path: Optional custom path to the parser executable
            temp_base_dir: Optional base directory for temp files (use short path to avoid Windows MAX_PATH limit)
        """
        self.site_name = site_name

        # Convert to list if single file
        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = file_paths

        # Use provided path or get from environment
        self.parser_exe_path = parser_exe_path

        # Setup working directory - use short prefix to avoid Windows MAX_PATH (260) limit
        self.work_dir = tempfile.mkdtemp(prefix='gw_', dir=temp_base_dir if temp_base_dir else None)
        
    def _validate_parser_executable(self) -> None:
        """Validate that the parser executable exists and is usable."""
        if not self.parser_exe_path:
            raise ValueError("Parser executable path is not set")
        
        parser_path = Path(self.parser_exe_path)
        
        if not parser_path.exists():
            raise FileNotFoundError(
                f"Gateway parser executable not found at: {self.parser_exe_path}"
            )
        
        if not parser_path.is_file():
            raise ValueError(
                f"Gateway parser path is not a file: {self.parser_exe_path}"
            )
        
        if not os.access(self.parser_exe_path, os.R_OK):
            raise PermissionError(
                f"Gateway parser executable is not readable: {self.parser_exe_path}"
            )

    def _validate_input_files(self) -> None:
        """Validate that all input files exist."""
        if not self.file_paths:
            raise ValueError("No input files provided")
        
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")

    def _create_commands(self) -> List[str]:
        """Create command list for subprocess execution (deprecated - use _run_parser_all_files instead)."""
        # Use command format from parse_gateway_messages.py:
        # --sitename="test" --files="file1 file2 file3"
        files_str = " ".join(self.file_paths)
        commands = [
            self.parser_exe_path,
            f'--sitename={self.site_name}',
            f'--files={files_str}'
        ]
        
        return commands

    def run_parser(self, max_retries: int = 3, timeout: int = 600) -> Dict[str, Any]:
        """
        Execute the gateway parser and return parsed data.
        
        Processes all files together in a single command (following parse_gateway_messages.py logic).

        Args:
            max_retries: Maximum number of retries
            timeout: Timeout in seconds (default 10 minutes for large files)

        Returns:
            Dictionary containing parsed gateway data or empty dict on error
        """
        try:
            # Validate prerequisites
            self._validate_parser_executable()
            self._validate_input_files()
            
            # Process all files together (following parse_gateway_messages.py logic)
            return self._run_parser_all_files(max_retries, timeout)
                    
        except Exception as e:
            return {"error": str(e)}
        finally:
            # Cleanup work directory
            try:
                import shutil
                if os.path.exists(self.work_dir):
                    shutil.rmtree(self.work_dir, ignore_errors=True)
            except Exception:
                pass
    
    def _run_parser_all_files(self, max_retries: int, timeout: int) -> Dict[str, Any]:
        """Run parser for all files together (following parse_gateway_messages.py logic)."""
        # Use absolute paths
        abs_paths = [os.path.abspath(fp) for fp in self.file_paths]

        # Split into batches based on:
        # 1. MAX_FILES_PER_BATCH to prevent memory issues
        # 2. Windows command line limit (~32767 chars, use 30000 as safe limit)
        max_cmd_len = 30000
        base_cmd_len = len(self.parser_exe_path) + len(f'--sitename={self.site_name}') + 20

        batches = []
        current_batch = []
        current_len = base_cmd_len

        for fp in abs_paths:
            fp_len = len(fp) + 1  # +1 for space
            # Start new batch if: command line too long OR file count exceeds limit
            if current_batch and (current_len + fp_len > max_cmd_len or len(current_batch) >= MAX_FILES_PER_BATCH):
                batches.append(current_batch)
                current_batch = [fp]
                current_len = base_cmd_len + fp_len
            else:
                current_batch.append(fp)
                current_len += fp_len

        if current_batch:
            batches.append(current_batch)

        total_batches = len(batches)
        total_files = len(abs_paths)
        print(f"\n[Parser] Processing {total_files} files in {total_batches} batch(es)")

        # If only one batch, run normally
        if total_batches == 1:
            print(f"[Parser] Parsing {len(batches[0])} files...")
            return self._run_parser_batch(batches[0], max_retries, timeout)

        # Multiple batches - merge incrementally to avoid memory spike
        merged_result = {}
        with tqdm(total=total_files, desc="Parsing files", unit="file") as pbar:
            for batch_idx, batch in enumerate(batches):
                result = self._run_parser_batch(batch, max_retries, timeout)
                if "error" in result:
                    return {"error": f"Batch {batch_idx + 1}/{total_batches} failed: {result['error']}"}

                # Merge incrementally
                merged_result = self._merge_two_results(merged_result, result)

                # Update progress
                pbar.update(len(batch))

                # Free memory after each batch
                del result
                gc.collect()

        return merged_result

    def _run_parser_batch(self, file_paths: List[str], max_retries: int, timeout: int) -> Dict[str, Any]:
        """Run parser for a batch of files."""
        files_str = " ".join(file_paths)
        commands = [
            self.parser_exe_path,
            f'--sitename={self.site_name}',
            f'--files={files_str}'
        ]

        # Execute parser with retry logic
        attempt = 0
        while attempt < max_retries:
            attempt += 1

            try:
                result = subprocess.run(
                    commands,
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=self.work_dir,
                    timeout=timeout
                )

                if result.returncode == 0:
                    # Parse JSON from stderr (following parse_gateway_messages.py)
                    return self._read_parser_output(result)
                else:
                    if attempt >= max_retries:
                        return {
                            "error": f"Parser failed with return code {result.returncode}",
                            "stderr": result.stderr[:1000] if result.stderr else ""
                        }

                    # Retry with exponential backoff
                    sleep_duration = min(2 ** (attempt - 1), 10)
                    time.sleep(sleep_duration)
                    continue

            except subprocess.TimeoutExpired:
                if attempt >= max_retries:
                    return {"error": f"Parser execution timed out after {timeout} seconds"}
                continue
            except Exception as e:
                if attempt >= max_retries:
                    return {"error": str(e)}
                continue

        return {"error": "Max retries exceeded"}
    
    def _run_parser_single(self, file_path: str, max_retries: int, timeout: int) -> Dict[str, Any]:
        """Run parser for a single file."""
        # Use absolute path and normalize
        abs_path = os.path.abspath(file_path)
        
        # Use command format from parse_gateway_messages.py
        commands = [
            self.parser_exe_path,
            f'--sitename={self.site_name}',
            f'--files={abs_path}'
        ]
        
        # Execute parser with retry logic
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            
            try:
                result = subprocess.run(
                    commands,
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=self.work_dir,
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    # Parse JSON from stderr
                    return self._read_parser_output(result)
                else:
                    if attempt >= max_retries:
                        return {
                            "error": f"Parser failed with return code {result.returncode}",
                            "stderr": result.stderr[:1000] if result.stderr else ""
                        }
                    
                    # Retry with exponential backoff
                    sleep_duration = min(2 ** (attempt - 1), 10)
                    time.sleep(sleep_duration)
                    continue
                    
            except subprocess.TimeoutExpired:
                if attempt >= max_retries:
                    return {"error": f"Parser execution timed out after {timeout} seconds"}
                continue
            except Exception as e:
                if attempt >= max_retries:
                    return {"error": str(e)}
                continue
        
        return {"error": "Max retries exceeded"}
    
    def _merge_two_results(self, merged: Dict[str, Any], new_result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new_result into merged dict incrementally."""
        if not merged:
            return new_result
        if not new_result:
            return merged

        for key, value in new_result.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(merged[key], list) and isinstance(value, list):
                merged[key].extend(value)
            elif isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                # If conflict, keep first value or convert to list
                if not isinstance(merged[key], list):
                    merged[key] = [merged[key]]
                if value not in merged[key]:
                    merged[key].append(value)

        return merged

    def _merge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple parser results into one (deprecated, use _merge_two_results)."""
        if not results:
            return {}

        merged = {}
        for result in results:
            merged = self._merge_two_results(merged, result)

        return merged

    def _read_parser_output(self, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """Read and parse the JSON output from the parser."""
        try:
            # Parser outputs JSON to stderr
            output = result.stderr.strip()
            if not output:
                return {"error": "Parser returned empty output"}
            
            return json.loads(output)
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to decode parser output: {str(e)}",
                "raw_output": result.stderr[:500] if result.stderr else ""
            }
        except Exception as e:
            return {"error": f"Error reading parser output: {str(e)}"}


def parse_gateway_files(
    site_name: str,
    file_paths: Union[str, List[str]],
    parser_exe_path: Optional[str] = None,
    temp_base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to parse gateway files.

    Args:
        site_name: Site name for the gateway data
        file_paths: Path to file or list of file paths to process
        parser_exe_path: Optional custom path to the parser executable
        temp_base_dir: Optional base directory for temp files (use short path to avoid Windows MAX_PATH limit)

    Returns:
        Dictionary containing parsed gateway data
    """
    parser = GatewayParserWrapper(
        site_name=site_name,
        file_paths=file_paths,
        parser_exe_path=parser_exe_path,
        temp_base_dir=temp_base_dir
    )
    return parser.run_parser()
