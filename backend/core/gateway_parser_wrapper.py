"""
Gateway Parser Wrapper

Wrapper for parsing gateway message files using GWMReader executable.
Adapted from gateway_parser.py for use in webapp backend.
"""

import json
import os
import subprocess
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


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
        parser_exe_path: Optional[str] = None
    ):
        """
        Initialize the Gateway Parser Wrapper.

        Args:
            site_name: Site name for the gateway data
            file_paths: Path to file or list of file paths to process
            parser_exe_path: Optional custom path to the parser executable
        """
        self.site_name = site_name
        
        # Convert to list if single file
        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = file_paths
        
        # Use provided path or get from environment
        self.parser_exe_path = parser_exe_path
        
        # Setup working directory
        self.work_dir = tempfile.mkdtemp(prefix='gateway_parser_')
        
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
        
        # Use command format from parse_gateway_messages.py
        files_str = " ".join(abs_paths)
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
    
    def _merge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple parser results into one."""
        if not results:
            return {}
        
        if len(results) == 1:
            return results[0]
        
        # Try to merge JSON structures
        # This is a simple merge - may need adjustment based on actual data structure
        merged = {}
        for result in results:
            if isinstance(result, dict):
                for key, value in result.items():
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
    parser_exe_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to parse gateway files.

    Args:
        site_name: Site name for the gateway data
        file_paths: Path to file or list of file paths to process
        parser_exe_path: Optional custom path to the parser executable

    Returns:
        Dictionary containing parsed gateway data
    """
    parser = GatewayParserWrapper(
        site_name=site_name,
        file_paths=file_paths,
        parser_exe_path=parser_exe_path
    )
    return parser.run_parser()
