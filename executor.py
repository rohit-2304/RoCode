from typing import Tuple
from validator import validate_code
import tempfile
import subprocess
import os

def execute_code(code: str) -> Tuple[bool, str]:
    # validate first
    is_safe, errors = validate_code(code)
    if not is_safe:
        return False, "Validation failed:\n" + "\n".join(errors)

    # write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            text=True,
            timeout=10,
            env={
                'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
                'HOME': '/tmp',
            }
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Timeout: Code took too long to execute"
    finally:
        os.unlink(temp_path)