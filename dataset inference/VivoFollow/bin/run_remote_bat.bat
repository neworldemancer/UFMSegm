@echo off 


pushd %~dp0 
:loop 

IF exist remote.bat (
  call remote.bat
  del remote.bat
) ELSE (
  ping -n 5 127.0.0.1  >nul 2>&1
)
GOTO loop

popd