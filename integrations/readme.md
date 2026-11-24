## To install SuperNova in the python environment
#### 1. Capture the active environment's Python executable
$pythonPath = python -c "import sys; print(sys.executable)"

#### 2. Build SuperNova against that interpreter
powershell -NoProfile -ExecutionPolicy Bypass -File .\SuperNova-Windows-Install.ps1 -BuildPython -PythonExecutable $pythonPath

#### 3. Install the updated bindings and Python integrations
python .\integrations\InstallSuperNovaBindings.py "build-acpp-windows"