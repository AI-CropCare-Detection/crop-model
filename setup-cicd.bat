@echo off
REM CI/CD Setup Script for Railway Deployment (Windows)
REM This script helps configure GitHub Actions for automatic Railway deployment

setlocal enabledelayedexpansion

echo [INFO] Railway CI/CD Setup Script
echo [INFO] ==============================
echo.

REM Step 1: Check prerequisites
echo [STEP 1] Checking prerequisites...

where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not installed
    exit /b 1
)
echo [OK] Git found

for /f %%i in ('git rev-parse --git-dir 2^>nul') do set GIT_DIR=%%i
if "%GIT_DIR%"=="" (
    echo [ERROR] Not in a git repository
    exit /b 1
)
echo [OK] Git repository detected

REM Step 2: Verify workflow files exist
echo.
echo [STEP 2] Verifying workflow files...

set WORKFLOWS_DIR=.github\workflows
if not exist "%WORKFLOWS_DIR%" (
    mkdir "%WORKFLOWS_DIR%"
    echo [OK] Created %WORKFLOWS_DIR% directory
) else (
    echo [OK] Workflows directory exists
)

if exist "%WORKFLOWS_DIR%\railway-deploy.yml" (
    echo [OK] railway-deploy.yml exists
) else (
    echo [WARN] railway-deploy.yml not found
)

if exist "%WORKFLOWS_DIR%\validate-pr.yml" (
    echo [OK] validate-pr.yml exists
) else (
    echo [WARN] validate-pr.yml not found
)

REM Step 3: Verify project structure
echo.
echo [STEP 3] Verifying project structure...

setlocal enabledelayedexpansion
for %%F in (main.py requirements.txt Dockerfile railway.json) do (
    if exist "%%F" (
        echo [OK] %%F exists
    ) else (
        echo [ERROR] %%F not found
        exit /b 1
    )
)

if exist "checkpoints" (
    echo [OK] checkpoints\ directory exists
    
    if exist "checkpoints\model_meta.json" (
        echo [OK] model_meta.json found
    ) else (
        echo [WARN] model_meta.json not found
    )
    
    if exist "checkpoints\yolov7_plant_disease.torchscript.pt" (
        echo [OK] TorchScript model found
    ) else (
        echo [WARN] TorchScript model not found
    )
) else (
    echo [ERROR] checkpoints\ directory not found
    exit /b 1
)

REM Step 4: Summary and next steps
echo.
echo ==============================
echo [SUCCESS] Setup verification complete!
echo ==============================
echo.
echo Next Steps:
echo 1. Go to GitHub Repository Settings
echo 2. Navigate to: Secrets and variables ^-^> Actions
echo 3. Add new secret: RAILWAY_TOKEN
echo    - Get token from: https://railway.app ^-^> Account Settings ^-^> Tokens
echo 4. Commit and push your changes:
echo    git add .
echo    git commit -m "ci: add Railway CI/CD pipeline"
echo    git push origin main
echo 5. Go to Actions tab and watch the deployment
echo.
echo Useful Commands:
echo   * View workflow runs: Check GitHub Actions tab
echo   * Manual deploy: railway up --detach
echo   * Check logs: railway logs
echo   * Test locally: python test_api.py
echo.
echo Documentation:
echo   * See DEPLOYMENT_CI_CD.md for detailed guide
echo   * See railway.json for deployment config
echo   * See Dockerfile for container config
echo.
pause
