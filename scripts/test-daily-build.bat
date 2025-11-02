@echo off
echo Testing daily build script...
echo Current directory: %CD%
echo Python path test: "C:\Users\bimal\AppData\Local\Programs\Python\Python311\python.exe" --version
"C:\Users\bimal\AppData\Local\Programs\Python\Python311\python.exe" --version

echo.
echo Testing module builder script...
"C:\Users\bimal\AppData\Local\Programs\Python\Python311\python.exe" scripts/build_module_descriptions.py --help

echo.
echo Test completed at %date% %time%