@echo off
setlocal

set vswherepath="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set vcvarsall_arch="64"

for /f "usebackq delims=" %%i in (`%vswherepath% -latest -property installationPath`) do (
  if exist "%%i\VC\Auxiliary\Build\vcvars%vcvarsall_arch%.bat" (
    set "vcvarsall=%%i\VC\Auxiliary\Build\vcvars%vcvarsall_arch%.bat"
  )
)

echo "Call vcvarsall.bat"
call "%vcvarsall%"

endlocal
