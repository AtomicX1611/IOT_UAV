# How To Run

## 1) Open project folder
In PowerShell:

```powershell
cd "D:\Documents\IOT\Project - Copy"
```

## 2) Activate virtual environment
Use whichever environment exists in your folder.

If you use `.venv`:

```powershell
& ".\.venv\Scripts\Activate.ps1"
```

If you use `venv`:

```powershell
& ".\venv\Scripts\Activate.ps1"
```

## 3) Install dependency (first time only)

```powershell
& ".\venv\Scripts\python.exe" -m pip install plotly
```

## 4) Run the simulation

```powershell
& ".\venv\Scripts\python.exe" .\main.py
```

Alternative (direct interpreter path):

```powershell
& ".\.venv\Scripts\python.exe" .\main.py
```

## 4.1) Quick environment check (recommended)

```powershell
py -c "import sys; print(sys.executable)"
```

If this path is not inside your project virtual environment, use one of the explicit interpreter commands above.

## 5) Output
- An interactive Plotly window opens in browser.
- HTML export is saved as `aco_3d_simulation.html` in the project root.

## Quick checks
If activation is blocked by policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again.
