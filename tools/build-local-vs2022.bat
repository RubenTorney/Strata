@echo off
rem Local engine build for perf A/B (VS 2022 Build Tools + CUDA 12.8, sm_86), same definitions as setup.py's CUDA 12 build.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CMAKE=C:\Users\urben\Documents\strata\.venv\Scripts\cmake.exe
set NINJA=C:\Users\urben\Documents\strata\.venv\Scripts\ninja.exe
set SRC=C:\Users\urben\Documents\strata-dev
set B=%SRC%\build-vs22
"%CMAKE%" -G Ninja -DCMAKE_MAKE_PROGRAM=%NINJA% -S %SRC% -B %B% -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_EXPERIMENTAL_SM60=ON "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler" "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_PATH%" -DSTRATA_GGML_DIR=C:\Users\urben\Documents\strata\third_party\llama.cpp || exit /b 1
"%CMAKE%" --build %B% --target strata -j 6 && exit /b 0
echo   (the build stopped - trying it once more)
"%CMAKE%" --build %B% --target strata -j 6 || exit /b 1
