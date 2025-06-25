# Installing PyBNesian 
Here you can find a detailed installation guide to use PyBNesian including the installation of C++ and GPU tools.

We acknowledge all the members from Computational Intelligence Group (UPM) for
further discussions related to the installation procedure.

### Contents
1. [Ubuntu and Linux sub-systems](#ubuntu-and-linux-sub-systems)
2. [Windows](#windows)
3. [Installation issues](#installation-issues)

## Ubuntu and Linux sub-systems
PyBNesian uses C++ and OpenCL in the backend to speed up certain computations. 
Thus, some software is required to ensure everything works. 
Note that, although setting up a Conda environment is usually recommended, it is not mandatory. 
The following commands ensure that the C++ and OpenCL requirements are satisfied.

```bash
sudo apt update
sudo apt install cmake
sudo apt install g++
sudo apt install opencl-headers 
sudo apt install ocl-icd-opencl-dev
```

After the previous steps you should be able to install PyBNesian and its dependencies.

### Installing from source
To install from source, we will download git to be able to download the
repository from GitHub.
```bash
sudo apt install git
```

Now, clone the repository, install its dependencies, and install the package. 

```bash
git clone https://github.com/carloslihu/PyBNesian.git
cd PyBNesian
pip install . --verbose
```

If you want to pre-compile the C++ code, you can use the following command.
This will create a wheel file in the `dist` folder, which can be used for installation
or distribution.
```bash
pip wheel . -w dist --verbose
pip install dist/pybnesian*.whl
```
### Installing directly from PyPi
Before installing PyBNesian, ensure that all the dependencies are already installed in your Python environment.

```bash
pip install PyBNesian
```

If no errors were raised, then the software is ready to be used. Otherwise, please
restart the process or raise an issue in the repository.

## Windows
Sometimes, in order to reduce possible inconvenient regarding Windows OS,
a Linux sub-system is installed (https://learn.microsoft.com/es-es/windows/wsl/install).
If this was the case, please go to [Ubuntu and Linux sub-systems](#ubuntu-and-linux-sub-systems) section.
Otherwise, please follow the next steps.

1. Download Visual Studio 2022 from https://visualstudio.microsoft.com/es/vs/ 
   
   1.1. Download the requirements for C++
3. Download Visual Studio Build Tools 2022.

```bash
winget install "Visual Studio Build Tools 2022"
```

3. Download developer tools for GPU. 

   3.1. For Nvidia, download Nvidia Toolkit (https://developer.nvidia.com/cuda-downloads)

   3.2. For Intel, download OneApi (https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)

5. Download OpenCL for windows. This guide explains the installation process: https://windowsreport.com/opencl-install-windows-11/

6. Install PyBNesian

### Installing from source
To install from source, we will download git to be able to download the
repository from GitHub.
```bash
sudo apt install git
```

Now, clone the repository, install its dependencies, and install the package. 

```bash
git clone https://github.com/carloslihu/PyBNesian.git
cd PyBNesian
pip install .
```

### Installing directly from PyPi
Before installing PyBNesian, ensure that all the dependencies are already installed in your Python environment.

```bash
pip install PyBNesian
```

If no errors were raised, then the software is ready to be used. 
Otherwise, please restart the process or raise an issue in the repository.

## Installation issues

1. If default [Ubuntu and Linux sub-systems](#ubuntu-and-linux-sub-systems) installation
fails, there might be necessary to install GPU toolkits for Linux. 
Please, visit https://developer.nvidia.com/cuda-downloads for Nvidia, and 
https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html for Intel.