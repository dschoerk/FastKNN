cd ..
rmdir /s /q build_vs2012_x64_test
mkdir build_vs2012_x64_test
cd build_vs2012_x64_test
cmake -G "Visual Studio 11 Win64" ..
call "%VS100COMNTOOLS%\..\..\VC\bin\vcvars32.bat"
"%VS110COMNTOOLS%\..\IDE\devenv.com" "fastknn.sln" /out build.log /build "Release" /project ALL_BUILD
