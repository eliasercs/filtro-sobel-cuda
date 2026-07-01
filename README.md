# Actividad 4

## Instrucciones de compilación

Navegar al directorio del código fuente
```
cd src
```

Compilar el programa
```
nvcc -std=c++17 image.cpp main.cpp -o secuencial
```

Ejecución
```
secuencial --instance=<small|medium|large|no-divisible> --kernel-size=<size>
```
size debe ser número impar mayor o igual a 3.