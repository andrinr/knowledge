# Web assembly

## Installtion

### Windows

```powershell
git clone https://github.com/emscripten-core/emsdk.git

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

./emsdk install latest

./emsdk activate latest

./emsdk_env.bat
```

## Compile

### WIndows

```{powershell}
./emsdk_env.bat
emcc ./<file>.cpp
emcc ./<file>.c -o out.html
```

