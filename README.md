<h1 align="center">BetaDev</h1>
<p align="center">AI Tools for Red Team Exploits</p>
<p align="center">
  <kbd>
    <img src="https://github.com/Tread-Stone/BetaDev/assets/61941978/e6791c97-f5f8-4ccb-8294-bfd127b3a0c5" width="500" height="auto"></img>
  </kbd>
</p>


## Overview
BetaDev is an ambitious project attempting to create an AlphaDev Language Server Protocol (LSP) for C/C++. This project includes a variety of components, from neural network implementations to big data processing scripts.

## Features
- **Neural Network Implementation**: A custom neural network library written in C, found in [lib/neuralnet.h](https://github.com/Tread-Stone/BetaDev/blob/main/lib/neuralnet.h).
- **Big Data Processing**: Python scripts for handling large datasets, including MNIST data processing in [mnist.py](https://github.com/Tread-Stone/BetaDev/blob/main/big_data/mnist.py) and a data preprocessor in [preprocessor.py](https://github.com/Tread-Stone/BetaDev/blob/main/big_data/preprocessor.py).
- **Automated Workflows**: GitHub Actions for automated building and testing, as seen in [.github/workflows/make-single-platform.yml](https://github.com/Tread-Stone/BetaDev/blob/main/.github/workflows/make-single-platform.yml).
- **Testing Suite**: A suite of tests to ensure code reliability, located in [tests/run-tests.cpp](https://github.com/Tread-Stone/BetaDev/blob/main/tests/run-tests.cpp).

## Getting Started
To test the project, follow these steps:
```bash
chmod +rx build.sh
./build.sh && ./nn
```

## Build Instructions
The project uses CMake for building. The `CMakeLists.txt` file is available [here](https://github.com/Tread-Stone/BetaDev/blob/main/CMakeLists.txt).

## Co-authors

Thanks to these wonderful people:

[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<table>
  <tr>
    <td align="center"><a href="https://github.com/BrendanGlancy"><img src="https://avatars.githubusercontent.com/u/61941978?v=4" width="100px;" alt=""/><br /><sub><b>Brendan Glancy</b></sub></a><br /><a title="Code">💻</a></a></td>
    <td align="center"><a href="https://github.com/HDTHREE"><img src="https://avatars.githubusercontent.com/u/98629093?v=4" width="100px;" alt=""/><br /><sub><b>Hayden Dennis</b></sub></a><br /><a title="Code">💻</a> </a></td>
    <td align="center"><a href="https://github.com/Coontzy1"><img src="https://avatars.githubusercontent.com/u/48108269?v=4" width="100px;" alt=""/><br /><sub><b>Austin Coontz</b></sub></a><br /><a title="Code">💻</a></td>
  </tr>
</table>

## License
This project is licensed under the [MIT License](https://github.com/Tread-Stone/BetaDev/blob/main/LICENSE).
