@echo off
setlocal

set MSVC_ROOT=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.39.33519
set MSVC_BIN=%MSVC_ROOT%\bin\Hostx64\x64
set MSVC_LIB=%MSVC_ROOT%\lib\x64
set MSVC_INC=%MSVC_ROOT%\include

set WINSDK_VER=10.0.26100.0
set WINSDK_BIN=C:\Program Files (x86)\Windows Kits\10\bin\%WINSDK_VER%\x64
set WINSDK_LIB_UM=C:\Program Files (x86)\Windows Kits\10\Lib\%WINSDK_VER%\um\x64
set WINSDK_LIB_UCRT=C:\Program Files (x86)\Windows Kits\10\Lib\%WINSDK_VER%\ucrt\x64
set WINSDK_INC_UM=C:\Program Files (x86)\Windows Kits\10\Include\%WINSDK_VER%\um
set WINSDK_INC_UCRT=C:\Program Files (x86)\Windows Kits\10\Include\%WINSDK_VER%\ucrt
set WINSDK_INC_SHARED=C:\Program Files (x86)\Windows Kits\10\Include\%WINSDK_VER%\shared

set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set CUDA_BIN=%CUDA_ROOT%\bin
set NINJA_BIN=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja

set PATH=%MSVC_BIN%;%WINSDK_BIN%;%CUDA_BIN%;%NINJA_BIN%;%PATH%
set LIB=%MSVC_LIB%;%WINSDK_LIB_UM%;%WINSDK_LIB_UCRT%;%LIB%
set INCLUDE=%MSVC_INC%;%WINSDK_INC_UM%;%WINSDK_INC_UCRT%;%WINSDK_INC_SHARED%;%CUDA_ROOT%\include;%INCLUDE%
set CUDA_PATH=%CUDA_ROOT%

echo [Build] Verifying tools...
where cl.exe
where nvcc.exe
where ninja.exe

echo.
echo [Build] Running CMake configure...
"C:\Program Files\CMake\bin\cmake.exe" -G Ninja -B build -S . ^
  -DCMAKE_MAKE_PROGRAM="%NINJA_BIN%\ninja.exe" ^
  -DCMAKE_C_COMPILER="%MSVC_BIN%\cl.exe" ^
  -DCMAKE_CXX_COMPILER="%MSVC_BIN%\cl.exe" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_BIN%\nvcc.exe" ^
  -DCUDAToolkit_ROOT="%CUDA_ROOT%" ^
  -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% NEQ 0 (
    echo [Build] CMake configure FAILED with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo [Build] Running CMake build...
"C:\Program Files\CMake\bin\cmake.exe" --build build

if %ERRORLEVEL% NEQ 0 (
    echo [Build] CMake build FAILED with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo [Build] SUCCESS! Running readiness report...
"C:\Users\rishi\AppData\Local\Programs\Python\Python311\python.exe" tests\readiness_report.py

endlocal
