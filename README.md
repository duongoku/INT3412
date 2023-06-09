# INT3412

An application for computer vision course

## Pre-requisites

-   [Python 3.10](https://www.python.org/downloads/)
-   [Pytorch with CUDA](https://pytorch.org/get-started/locally/)
-   A NVIDIA GPU with CUDA support (IDK if it works without CUDA, didn't test it)

## Installation

-   Create a virtual environment

```bash
python -m venv .env
```

-   Activate the virtual environment

```bash
source .env/bin/activate
```

-   Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

-   Create a `temp` folder in the root directory

```bash
mkdir temp
```

-   Run the application

```bash
python server.py
```

## Demonstration

![demo](assets/demo.gif)

## Acknowledgements

-   [Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD) for the implementation of Patch-NetVLAD, contents in the `patchnetvlad` directory and part of some other files are taken from this repository. For the sake of simplicity, I just pushed the whole `patchnetvlad` directory to this repository.
-   Various image sources for the images in the `image` directory.
-   Assoc. Prof. Dr. [Le Thanh Ha](https://uet.vnu.edu.vn/~ltha/CV.pdf) for the guidance and support throughout the course.

## Notes

-   The server response time is quite long, it may take up to several minutes to get the result.

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
