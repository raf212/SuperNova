### To install SuperNova in the python environment
powershell -NoProfile -ExecutionPolicy Bypass -File .\SuperNova-Windows-Install.ps1 -BuildPython -PythonExecutable $pythonPath

python .\integrations\InstallSuperNovaBindings.py "build-acpp-windows"