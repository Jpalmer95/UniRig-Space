import gradio as gr
import torch
import os
import sys
import tempfile
import shutil
import subprocess
import spaces # Keep this if you use @spaces.GPU
from typing import Any, Dict, List, Union # Added Union

# --- Configuration ---
APP_ROOT_DIR = os.path.abspath(os.path.dirname(__file__)) # Should be /home/user/app

UNIRIG_REPO_DIR = os.path.join(APP_ROOT_DIR, "UniRig")

BLENDER_VERSION_NAME = "blender-4.2.0-linux-x64"
BLENDER_LOCAL_INSTALL_BASE_DIR = os.path.join(APP_ROOT_DIR, "blender_installation")
BLENDER_INSTALL_DIR = os.path.join(BLENDER_LOCAL_INSTALL_BASE_DIR, BLENDER_VERSION_NAME)

BLENDER_PYTHON_VERSION_DIR = "4.2" # From Blender's internal structure
BLENDER_PYTHON_VERSION = "python3.11" # UniRig requirement

BLENDER_PYTHON_DIR = os.path.join(BLENDER_INSTALL_DIR, BLENDER_PYTHON_VERSION_DIR, "python")
BLENDER_PYTHON_BIN_DIR = os.path.join(BLENDER_PYTHON_DIR, "bin")
BLENDER_EXEC = os.path.join(BLENDER_INSTALL_DIR, "blender")

LOCAL_BIN_DIR = os.path.join(APP_ROOT_DIR, "local_bin")
BLENDER_EXEC_LOCAL_SYMLINK = os.path.join(LOCAL_BIN_DIR, "blender")
BLENDER_EXEC_SYMLINK = "/usr/local/bin/blender" # Fallback system symlink

SETUP_SCRIPT = os.path.join(APP_ROOT_DIR, "setup_blender.sh")
SETUP_SCRIPT_TIMEOUT = 1800

# --- Initial Checks ---
print("--- Environment Checks ---")
print(f"APP_ROOT_DIR: {APP_ROOT_DIR}")
print(f"Expected Blender Install Dir: {BLENDER_INSTALL_DIR}")
print(f"Expected Blender Executable: {BLENDER_EXEC}")

blender_executable_to_use = None
if os.path.exists(BLENDER_EXEC):
    print(f"Blender executable found at direct local path: {BLENDER_EXEC}")
    blender_executable_to_use = BLENDER_EXEC
elif os.path.exists(BLENDER_EXEC_LOCAL_SYMLINK):
    print(f"Blender executable found via local symlink: {BLENDER_EXEC_LOCAL_SYMLINK}")
    blender_executable_to_use = BLENDER_EXEC_LOCAL_SYMLINK
elif os.path.exists(BLENDER_EXEC_SYMLINK):
    print(f"Blender executable found via system symlink: {BLENDER_EXEC_SYMLINK}")
    blender_executable_to_use = BLENDER_EXEC_SYMLINK
else:
    print(f"Blender executable not found. Running setup script...")
    if os.path.exists(SETUP_SCRIPT):
        try:
            setup_result = subprocess.run(
                ["bash", SETUP_SCRIPT],
                check=True,
                capture_output=True,
                text=True,
                timeout=SETUP_SCRIPT_TIMEOUT
            )
            print("Setup script executed successfully.")
            print(f"Setup STDOUT:\n{setup_result.stdout}")
            if setup_result.stderr: print(f"Setup STDERR:\n{setup_result.stderr}")

            if os.path.exists(BLENDER_EXEC):
                blender_executable_to_use = BLENDER_EXEC
                print(f"Blender executable now found at direct local path: {BLENDER_EXEC}")
            elif os.path.exists(BLENDER_EXEC_LOCAL_SYMLINK):
                blender_executable_to_use = BLENDER_EXEC_LOCAL_SYMLINK
                print(f"Blender executable now found via local symlink: {BLENDER_EXEC_LOCAL_SYMLINK}")
            elif os.path.exists(BLENDER_EXEC_SYMLINK):
                 blender_executable_to_use = BLENDER_EXEC_SYMLINK
                 print(f"Blender executable now found via system symlink: {BLENDER_EXEC_SYMLINK}")

            if not blender_executable_to_use:
                 raise RuntimeError(f"Setup script ran but Blender executable still not found.")
        except subprocess.TimeoutExpired:
             print(f"ERROR: Setup script timed out: {SETUP_SCRIPT}")
             raise gr.Error(f"Setup script timed out. Check logs.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR running setup script: {SETUP_SCRIPT}\nStderr: {e.stderr}")
            raise gr.Error(f"Failed to execute setup script. Stderr: {e.stderr[-500:]}")
        except Exception as e:
            raise gr.Error(f"Unexpected error running setup script '{SETUP_SCRIPT}': {e}")
    else:
        raise gr.Error(f"Blender executable not found and setup script missing: {SETUP_SCRIPT}")

bpy_import_ok = False
if blender_executable_to_use:
    try:
        print("Testing bpy import via Blender...")
        test_script_content = "import bpy; print('bpy imported successfully')"
        test_result = subprocess.run(
            [blender_executable_to_use, "--background", "--python-expr", test_script_content],
            capture_output=True, text=True, check=True, timeout=30
        )
        if "bpy imported successfully" in test_result.stdout:
            print("Successfully imported 'bpy' using Blender executable.")
            bpy_import_ok = True
        else:
            print(f"WARNING: 'bpy' import test unexpected output:\nSTDOUT:{test_result.stdout}\nSTDERR:{test_result.stderr}")
    except subprocess.TimeoutExpired:
        print("WARNING: 'bpy' import test via Blender timed out.")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to import 'bpy' using Blender executable:\nSTDOUT:{e.stdout}\nSTDERR:{e.stderr}")
    except Exception as e:
        print(f"WARNING: Unexpected error during 'bpy' import test: {e}")
else:
     print("WARNING: Cannot test bpy import as Blender executable was not found.")

unirig_repo_ok = False
unirig_run_py_ok = False
UNIRIG_RUN_PY = os.path.join(UNIRIG_REPO_DIR, "run.py")
if not os.path.isdir(UNIRIG_REPO_DIR):
    raise gr.Error(f"UniRig repository missing at: {UNIRIG_REPO_DIR}.")
else:
    print(f"UniRig repository found at: {UNIRIG_REPO_DIR}")
    unirig_repo_ok = True
    if not os.path.exists(UNIRIG_RUN_PY):
        raise gr.Error(f"UniRig's run.py not found at {UNIRIG_RUN_PY}.")
    else:
        unirig_run_py_ok = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Gradio environment using device: {DEVICE}")
if DEVICE.type == 'cuda':
    try:
        print(f"Gradio CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Gradio PyTorch CUDA Built Version: {torch.version.cuda}")
    except Exception as e:
        print(f"Could not get Gradio CUDA device details: {e}")
else:
    print("Warning: Gradio environment CUDA not available.")
print("--- End Environment Checks ---")

def patch_asset_py():
    asset_py_path = os.path.join(UNIRIG_REPO_DIR, "src", "data", "asset.py")
    try:
        if not os.path.exists(asset_py_path):
            print(f"Warning: asset.py not found at {asset_py_path}, skipping patch.")
            return
        with open(asset_py_path, "r") as f: content = f.read()
        problematic_line = "meta: Union[Dict[str, ...], None]=None"
        corrected_line = "meta: Union[Dict[str, Any], None]=None"
        typing_import = "from typing import Any"
        if corrected_line in content:
            print("Patch already applied to asset.py"); return
        if problematic_line not in content:
            print("Problematic line not found in asset.py, patch might be unnecessary."); return
        print("Applying patch to asset.py...")
        content = content.replace(problematic_line, corrected_line)
        if typing_import not in content:
            if "from typing import" in content:
                 content = content.replace("from typing import", f"{typing_import}\nfrom typing import", 1)
            else:
                 content = f"{typing_import}\n{content}"
        with open(asset_py_path, "w") as f: f.write(content)
        print("Successfully patched asset.py")
    except Exception as e:
        print(f"ERROR: Failed to patch asset.py: {e}. Proceeding cautiously.")

@spaces.GPU
def run_unirig_command(python_script_path: str, script_args: List[str], step_name: str):
    if not blender_executable_to_use:
        raise gr.Error("Blender executable path not determined. Cannot run UniRig step.")

    process_env = os.environ.copy()
    # PYTHONPATH is set here, but Blender's Python might not fully utilize it for sys.path initialization
    # as expected. The bootstrap script below will directly manipulate sys.path.
    unirig_src_dir = os.path.join(UNIRIG_REPO_DIR, 'src')
    pythonpath_parts = [UNIRIG_REPO_DIR, unirig_src_dir, APP_ROOT_DIR] # Added APP_ROOT_DIR for good measure
    existing_pythonpath = process_env.get('PYTHONPATH', '')
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    process_env["PYTHONPATH"] = os.pathsep.join(filter(None, pythonpath_parts))
    print(f"Subprocess PYTHONPATH (for Blender env): {process_env['PYTHONPATH']}")

    blender_main_lib_path = os.path.join(BLENDER_INSTALL_DIR, "lib")
    blender_python_lib_path = os.path.join(BLENDER_PYTHON_DIR, "lib")
    ld_path_parts = []
    if os.path.exists(blender_main_lib_path): ld_path_parts.append(blender_main_lib_path)
    if os.path.exists(blender_python_lib_path): ld_path_parts.append(blender_python_lib_path)
    existing_ld_path = process_env.get('LD_LIBRARY_PATH', '')
    if existing_ld_path: ld_path_parts.append(existing_ld_path)
    if ld_path_parts:
        process_env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, ld_path_parts))
    print(f"Subprocess LD_LIBRARY_PATH: {process_env.get('LD_LIBRARY_PATH', 'Not set')}")

    if os.path.isdir(LOCAL_BIN_DIR):
        process_env["PATH"] = f"{LOCAL_BIN_DIR}{os.pathsep}{process_env.get('PATH', '')}"
    print(f"Subprocess PATH: {process_env.get('PATH', 'Not set')}")

    # --- Create a bootstrap script to set sys.path correctly inside Blender's Python ---
    bootstrap_content = f"""
import sys
import os
import runpy

# Path to the UniRig repository root
# This directory needs to be in sys.path for 'from src...' imports to work
unirig_repo_dir_abs = '{os.path.abspath(UNIRIG_REPO_DIR)}'

# Path to the actual Python script Blender should run (e.g., UniRig's run.py or a diagnostic script)
original_script_to_run = '{os.path.abspath(python_script_path)}'

print(f"[Bootstrap] Original sys.path: {{sys.path}}", file=sys.stderr)
print(f"[Bootstrap] Current working directory: {{os.getcwd()}}", file=sys.stderr)
print(f"[Bootstrap] UNIRIG_REPO_DIR to add: {{unirig_repo_dir_abs}}", file=sys.stderr)
print(f"[Bootstrap] Original script to run: {{original_script_to_run}}", file=sys.stderr)

# Ensure the UniRig repository directory is at the beginning of sys.path
if unirig_repo_dir_abs not in sys.path:
    sys.path.insert(0, unirig_repo_dir_abs)
    print(f"[Bootstrap] Modified sys.path: {{sys.path}}", file=sys.stderr)
else:
    print(f"[Bootstrap] UNIRIG_REPO_DIR already in sys.path.", file=sys.stderr)

# Blender passes arguments to the --python script after a '--' separator in the main command.
# sys.argv for this bootstrap script will be: [bootstrap_script_path, '--'] + original_script_args
# We need to reconstruct sys.argv for the 'original_script_to_run'.
try:
    separator_index = sys.argv.index('--')
    args_for_original_script = sys.argv[separator_index + 1:]
except ValueError:
    # This case should not happen if app.py always adds '--'
    args_for_original_script = []

sys.argv = [original_script_to_run] + args_for_original_script
print(f"[Bootstrap] Executing: {{original_script_to_run}} with sys.argv: {{sys.argv}}", file=sys.stderr)

# Change CWD to UNIRIG_REPO_DIR just before running the script, if not already set.
# The subprocess call in app.py should already set cwd=UNIRIG_REPO_DIR.
# This is a safeguard or for clarity.
if os.getcwd() != unirig_repo_dir_abs:
    print(f"[Bootstrap] Changing CWD from {{os.getcwd()}} to {{unirig_repo_dir_abs}}", file=sys.stderr)
    os.chdir(unirig_repo_dir_abs)

# Execute the original script
try:
    runpy.run_path(original_script_to_run, run_name='__main__')
except Exception as e_runpy:
    print(f"[Bootstrap] Error running '{{original_script_to_run}}' with runpy: {{e_runpy}}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise # Re-raise the exception to ensure the calling process sees failure
"""
    temp_bootstrap_file = None # Define outside try block for visibility in finally
    try:
        # Use a named temporary file that Blender can access
        temp_bootstrap_file = tempfile.NamedTemporaryFile(mode='w', delete=False, prefix="blender_bootstrap_", suffix=".py")
        temp_bootstrap_file.write(bootstrap_content)
        temp_bootstrap_file.close() # Close the file so Blender can open it
        bootstrap_script_path_for_blender = temp_bootstrap_file.name
        print(f"Using temporary bootstrap script: {bootstrap_script_path_for_blender}")

        cmd = [
            blender_executable_to_use,
            "--background",
            "--python", bootstrap_script_path_for_blender, # Blender executes our bootstrap script
            "--"  # Separator for Blender args vs. bootstrap/target script args
        ] + script_args # These args are passed to the bootstrap script, which then passes them to the target

        print(f"\n--- Running UniRig Step (via bootstrap): {step_name} ---")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=UNIRIG_REPO_DIR, # CWD for the Blender process
            capture_output=True,
            text=True,
            check=True,
            env=process_env,
            timeout=1800
        )
        print(f"{step_name} STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"{step_name} STDERR (Info/Warnings):\n{result.stderr}")
            stderr_lower = result.stderr.lower()
            if "error" in stderr_lower or "failed" in stderr_lower or "traceback" in stderr_lower:
                 # Check for specific bootstrap errors first
                if "[bootstrap] error running" in result.stderr.lower():
                    print(f"ERROR: Bootstrap script reported an error running the target script for {step_name}.")
                elif "module 'src' not found" in result.stderr.lower() or "no module named 'src'" in result.stderr.lower() : # Check after bootstrap
                    print(f"ERROR: 'src' module still not found for {step_name} even after bootstrap. Check sys.path in logs.")
                else:
                    print(f"WARNING: Potential error messages found in STDERR for {step_name} despite success exit code.")

    except subprocess.TimeoutExpired:
        print(f"ERROR: {step_name} timed out after 30 minutes.")
        raise gr.Error(f"Processing step '{step_name}' timed out.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR during {step_name}: Subprocess failed!")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        # Print full stdout/stderr from the error for more context
        print(f"--- {step_name} STDOUT (on error) ---:\n{e.stdout}")
        print(f"--- {step_name} STDERR (on error) ---:\n{e.stderr}")
        
        error_summary = e.stderr.strip().splitlines()
        last_lines = "\n".join(error_summary[-25:]) if error_summary else "No stderr output." # Increased lines
        specific_error = "Unknown error."
        if "module 'src' not found" in e.stderr.lower() or "no module named 'src'" in e.stderr.lower() :
             specific_error = "UniRig script failed to import 'src' module. Bootstrap sys.path modification might have failed. Check diagnostic logs for sys.path content within Blender's Python."
        elif "ModuleNotFoundError: No module named 'bpy'" in e.stderr:
            specific_error = "The 'bpy' module could not be imported. Check Blender Python setup."
        # ... (other specific error checks remain the same) ...
        else:
            specific_error = f"Check logs. Last error lines:\n{last_lines}"
        raise gr.Error(f"Error in UniRig '{step_name}'. {specific_error}")
    except FileNotFoundError:
        print(f"ERROR: Could not find Blender executable or script for {step_name}.")
        raise gr.Error(f"Setup error for UniRig '{step_name}'. Files not found.")
    except Exception as e_general:
        print(f"An unexpected Python exception in run_unirig_command for {step_name}: {e_general}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Unexpected Python error during '{step_name}': {str(e_general)[:500]}")
    finally:
        # Clean up the temporary bootstrap script
        if temp_bootstrap_file and os.path.exists(temp_bootstrap_file.name):
            try:
                os.remove(temp_bootstrap_file.name)
                print(f"Cleaned up temporary bootstrap script: {temp_bootstrap_file.name}")
            except Exception as cleanup_e:
                print(f"Error cleaning up bootstrap script {temp_bootstrap_file.name}: {cleanup_e}")
    print(f"--- Finished UniRig Step (via bootstrap): {step_name} ---")


@spaces.GPU
def rig_glb_mesh_multistep(input_glb_file_obj):
    if not blender_executable_to_use:
        gr.Warning("System not ready: Blender executable not found.")
        return None
    if not unirig_repo_ok or not unirig_run_py_ok:
         gr.Warning("System not ready: UniRig repository or run.py script not found.")
         return None
    if not bpy_import_ok:
         gr.Warning("System warning: Initial 'bpy' import test failed. Proceeding cautiously.")

    try:
        patch_asset_py()
    except Exception as e:
        print(f"Ignoring patch error: {e}")

    if input_glb_file_obj is None:
        gr.Info("Please upload a .glb file first.")
        return None
    input_glb_path = input_glb_file_obj
    print(f"Input GLB path received: {input_glb_path}")

    if not isinstance(input_glb_path, str) or not os.path.exists(input_glb_path):
         raise gr.Error(f"Invalid input file path or file does not exist: {input_glb_path}")
    if not input_glb_path.lower().endswith(".glb"):
         raise gr.Error("Invalid file type. Please upload a .glb file.")

    # Use a single temporary directory for all processing for this run
    # This directory will be cleaned up at the end.
    # The bootstrap script will be created inside this if run_unirig_command doesn't make its own.
    # For clarity, run_unirig_command now handles its own bootstrap temp file.
    processing_temp_dir = tempfile.mkdtemp(prefix="unirig_processing_")
    print(f"Using temporary processing directory: {processing_temp_dir}")
    
    diagnostic_script_path = None # For cleanup in finally

    try:
        base_name = os.path.splitext(os.path.basename(input_glb_path))[0]
        abs_input_glb_path = os.path.abspath(input_glb_path)
        abs_skeleton_output_path = os.path.join(processing_temp_dir, f"{base_name}_skeleton.fbx")
        abs_skin_output_path = os.path.join(processing_temp_dir, f"{base_name}_skin.fbx")
        abs_final_rigged_glb_path = os.path.join(processing_temp_dir, f"{base_name}_rigged_final.glb")
        unirig_script_to_run = UNIRIG_RUN_PY # This is an absolute path

        print("\n--- Running Blender Python Environment Diagnostic Test (via bootstrap) ---")
        # Content of the diagnostic script remains the same
        diagnostic_script_content = f"""
import sys
import os
import traceback

print("--- Enhanced Diagnostic Info from Blender Python ---")
print(f"Python Executable: {{sys.executable}}")
print(f"Python Version: {{sys.version.replace('\n', ' ')}}") # Added sys.version
print(f"Current Working Directory (inside script): {{os.getcwd()}}")

print("\nsys.path:")
for i, p in enumerate(sys.path): print(f"  {{i}}: {{p}}")

print("\nPYTHONPATH Environment Variable (as seen by script):")
print(os.environ.get('PYTHONPATH', 'PYTHONPATH not set or empty'))

print("\nLD_LIBRARY_PATH Environment Variable (as seen by script):")
print(os.environ.get('LD_LIBRARY_PATH', 'LD_LIBRARY_PATH not set or empty'))

print("\n--- Attempting Critical Imports ---")

# 1. bpy
print("\n1. Attempting 'bpy' import...")
try:
    import bpy
    print("  SUCCESS: 'bpy' imported.")
    print(f"     bpy version: {{bpy.app.version_string}}")
except Exception as e:
    print(f"  FAILED to import 'bpy': {{e}}")
    traceback.print_exc(file=sys.stderr)

# 2. UniRig 'src' module
print("\n2. Checking for UniRig 'src' module availability...")
# UNIRIG_REPO_DIR from app.py's function scope will be interpolated here by Python when app.py runs
# The f-string formatting {os.path.abspath(UNIRIG_REPO_DIR)} ensures this path is correctly embedded
# into the script content that Blender's Python will execute.
print(f"   Expected UniRig repo parent in sys.path: '{os.path.abspath(UNIRIG_REPO_DIR)}'")
found_unirig_in_sys_path = any(os.path.abspath(UNIRIG_REPO_DIR) == os.path.abspath(p) for p in sys.path)
print(f"   Is UNIRIG_REPO_DIR ('{os.path.abspath(UNIRIG_REPO_DIR)}') in sys.path (at diagnostic script generation time)? {'Yes' if found_unirig_in_sys_path else 'No'}")


unirig_src_dir_in_cwd_exists = os.path.isdir('src')
print(f"   Is 'src' directory present in CWD ('{{os.getcwd()}}')? {'Yes' if unirig_src_dir_in_cwd_exists else 'No'}")
if unirig_src_dir_in_cwd_exists:
    init_py_in_src_exists = os.path.isfile(os.path.join('src', '__init__.py'))
    print(f"     Is 'src/__init__.py' present? {'Yes' if init_py_in_src_exists else 'No'}")

print("   Attempting 'from src.inference.download import download'...")
try:
    from src.inference.download import download
    print("  SUCCESS: 'from src.inference.download import download' worked.")
except ImportError as e:
    print(f"  FAILED: 'from src.inference.download import download': {{e}}")
    print(f"  Make sure '{os.path.abspath(UNIRIG_REPO_DIR)}' is correctly added to sys.path by the bootstrap script executed by Blender.")
    traceback.print_exc(file=sys.stderr)
except Exception as e:
    print(f"  FAILED: 'from src.inference.download import download' with other error: {{e}}")
    traceback.print_exc(file=sys.stderr)

# 3. flash_attn
print("\n3. Attempting 'flash_attn' import...")
try:
    import flash_attn
    print("  SUCCESS: 'flash_attn' imported.")
    if hasattr(flash_attn, '__version__'):
        print(f"     flash_attn version: {{flash_attn.__version__}}")
except Exception as e:
    print(f"  FAILED to import 'flash_attn': {{e}}")
    print(f"     Note: flash-attn is expected to be installed by setup_blender.sh from a specific wheel.")
    traceback.print_exc(file=sys.stderr)

# 4. spconv
print("\n4. Attempting 'spconv' import...")
try:
    import spconv
    print("  SUCCESS: 'spconv' imported.")
    if hasattr(spconv, 'constants') and hasattr(spconv.constants, 'SPCONV_VERSION'):
        print(f"     spconv version: {{spconv.constants.SPCONV_VERSION}}")
    elif hasattr(spconv, '__version__'):
        print(f"     spconv version: {{spconv.__version__}}")
except Exception as e:
    print(f"  FAILED to import 'spconv': {{e}}")
    print(f"     Note: spconv (e.g., spconv-cu118) should be installed via unirig_requirements.txt in Blender's Python.")
    traceback.print_exc(file=sys.stderr)

# 5. torch with CUDA check
print("\n5. Attempting 'torch' import and CUDA check...")
try:
    import torch
    print("  SUCCESS: 'torch' imported.")
    print(f"     torch version: {{torch.__version__}}")
    cuda_available = torch.cuda.is_available()
    print(f"     torch.cuda.is_available(): {{cuda_available}}")
    if cuda_available:
        print(f"       torch.version.cuda: {{torch.version.cuda}}")
        print(f"       torch.cuda.get_device_name(0): {{torch.cuda.get_device_name(0)}}")
        print(f"       torch.cuda.get_device_capability(0): {{torch.cuda.get_device_capability(0)}}")
    else:
        print(f"       CUDA not available to PyTorch in this Blender Python environment.")
        if "cpu" in torch.__version__: # Check if it's a CPU build explicitly
            print("       PyTorch build appears to be CPU-only.")
        else:
            print("       PyTorch build is not CPU-only, but CUDA is still not available. Check drivers/runtime/setup for Blender Python env.")

except Exception as e:
    print(f"  FAILED to import 'torch' or perform CUDA checks: {{e}}")
    traceback.print_exc(file=sys.stderr)

print("\n--- End Enhanced Diagnostic Info ---")
"""
        # Save diagnostic script to the processing_temp_dir
        diagnostic_script_path = os.path.join(processing_temp_dir, "env_diagnostic_test.py")
        with open(diagnostic_script_path, "w") as f: f.write(diagnostic_script_content)
        
        run_unirig_command(diagnostic_script_path, [], "Blender Env Diagnostic") # Args are empty for diagnostic
        print("--- Finished Blender Python Environment Diagnostic Test ---\n")
        # If the above didn't raise an error, sys.path was likely fixed by bootstrap for the diagnostic.

        unirig_device_arg = "device=cpu"
        if DEVICE.type == 'cuda':
             unirig_device_arg = "device=cuda:0"
        print(f"UniRig steps will attempt to use device argument: {unirig_device_arg}")

        print("\nStarting Step 1: Predicting Skeleton...")
        skeleton_args = [
            "--config-name=skeleton_config", "with",
            f"input={abs_input_glb_path}",
            f"output={abs_skeleton_output_path}",
            unirig_device_arg
        ]
        run_unirig_command(unirig_script_to_run, skeleton_args, "Skeleton Prediction")
        if not os.path.exists(abs_skeleton_output_path):
            raise gr.Error("Skeleton prediction failed. Output file not created.")
        print("Step 1: Skeleton Prediction completed.")

        print("\nStarting Step 2: Predicting Skinning Weights...")
        skin_args = [
            "--config-name=skin_config", "with",
            f"input={abs_skeleton_output_path}",
            f"output={abs_skin_output_path}",
            unirig_device_arg
        ]
        run_unirig_command(unirig_script_to_run, skin_args, "Skinning Prediction")
        if not os.path.exists(abs_skin_output_path):
            raise gr.Error("Skinning prediction failed. Output file not created.")
        print("Step 2: Skinning Prediction completed.")

        print("\nStarting Step 3: Merging Results...")
        merge_args = [
            "--config-name=merge_config", "with",
            f"source_path={abs_skin_output_path}",
            f"target_path={abs_input_glb_path}",
            f"output_path={abs_final_rigged_glb_path}",
            "mode=skin",
            unirig_device_arg
        ]
        run_unirig_command(unirig_script_to_run, merge_args, "Merging Results")
        if not os.path.exists(abs_final_rigged_glb_path):
            raise gr.Error("Merging process failed. Final rigged GLB file not created.")
        print("Step 3: Merging completed.")

        print(f"Successfully generated rigged model: {abs_final_rigged_glb_path}")
        return gr.update(value=abs_final_rigged_glb_path)
    except gr.Error as e:
        print(f"A Gradio Error occurred: {e}")
        # No need to re-raise, run_unirig_command already does or it's an explicit raise here.
        # Let Gradio handle displaying it.
        raise # Re-raise to ensure Gradio UI shows the error
    except Exception as e:
        print(f"An unexpected error occurred in rig_glb_mesh_multistep: {e}")
        import traceback; traceback.print_exc()
        raise gr.Error(f"An unexpected error occurred: {str(e)[:500]}. Check logs.")
    finally:
        # Cleanup diagnostic script if it was created
        if diagnostic_script_path and os.path.exists(diagnostic_script_path):
            try:
                os.remove(diagnostic_script_path)
                print(f"Cleaned up diagnostic script: {diagnostic_script_path}")
            except Exception as cleanup_e:
                print(f"Error cleaning up diagnostic script {diagnostic_script_path}: {cleanup_e}")
        # Cleanup the main processing directory
        if os.path.exists(processing_temp_dir):
             try:
                 shutil.rmtree(processing_temp_dir)
                 print(f"Cleaned up temp dir: {processing_temp_dir}")
             except Exception as cleanup_e:
                 print(f"Error cleaning up temp dir {processing_temp_dir}: {cleanup_e}")

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.sky,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

startup_error_message = None
if not blender_executable_to_use:
     startup_error_message = (f"CRITICAL STARTUP ERROR: Blender executable not located. Expected at {BLENDER_EXEC}")
elif not unirig_repo_ok:
     startup_error_message = (f"CRITICAL STARTUP ERROR: UniRig repository not found at {UNIRIG_REPO_DIR}.")
elif not unirig_run_py_ok:
      startup_error_message = (f"CRITICAL STARTUP ERROR: UniRig run.py not found at {UNIRIG_RUN_PY}.")

if startup_error_message:
    print(startup_error_message)
    with gr.Blocks(theme=theme) as iface:
        gr.Markdown(f"# Application Startup Error\n\n{startup_error_message}\n\nPlease check Space logs.")
else:
    with gr.Blocks(theme=theme) as iface:
        gr.Markdown(
             f"""
             # UniRig Auto-Rigger (Blender {BLENDER_PYTHON_VERSION_DIR} / Python {BLENDER_PYTHON_VERSION})
             Upload a `.glb` mesh. UniRig predicts skeleton and skinning weights via Blender's Python.
             * App Python: `{sys.version.split()[0]}`, UniRig (Blender): `{BLENDER_PYTHON_VERSION}`.
             * Device: **{DEVICE.type.upper()}**.
             * Blender: `{blender_executable_to_use}`.
             * UniRig Source: [https://github.com/VAST-AI-Research/UniRig](https://github.com/VAST-AI-Research/UniRig)
             """
         )
        with gr.Row():
            with gr.Column(scale=1):
                input_model = gr.File(
                    label="Upload .glb Mesh File",
                    type="filepath",
                    file_types=[".glb"]
                )
                submit_button = gr.Button("Rig Model", variant="primary")
            with gr.Column(scale=2):
                output_model = gr.Model3D(
                    label="Rigged 3D Model (.glb)",
                    clear_color=[0.8, 0.8, 0.8, 1.0],
                )
        submit_button.click(
            fn=rig_glb_mesh_multistep,
            inputs=[input_model],
            outputs=[output_model]
        )

if __name__ == "__main__":
    if 'iface' in locals():
        print("Launching Gradio interface...")
        iface.launch(share=False, ssr_mode=False)
    else:
        print("ERROR: Gradio interface not created due to startup errors. Check logs.")

