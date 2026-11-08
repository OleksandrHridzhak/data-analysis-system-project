@echo off
cd /d "%~dp0"
if not exist "latex_pdf" mkdir "latex_pdf"
set "TECTONIC=..\..\knowladge\bin\tectonic.exe"
if not exist "%TECTONIC%" set "TECTONIC=..\..\..\bin\tectonic.exe"
if not exist "%TECTONIC%" set "TECTONIC=c:\Users\userstudent\uni\knowladge\bin\tectonic.exe"
for %%f in (*.tex) do "%TECTONIC%" "%%f" --outdir latex_pdf
echo.
echo ============================================================
echo  Ready! PDF in latex_pdf\
echo ============================================================

