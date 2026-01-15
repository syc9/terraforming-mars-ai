for running on Windows

in powershell
1. open terminal at directory
2. run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
3. run: ./venv/Scripts/activate
4. run: uv run _______.py

for installing packages, add in pyproject.toml the dependencies needed
1. run: uv pip install .