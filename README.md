# Nonmanifold Minimal Surfaces
## Team
- **Fellows**: Alice Mehalek, Natasha Diederen, Zhecheng Wang, Zeltzyn Guadalupe Montes Rosales, Olga Guțan
- **TA**: Erik Amezquita
- **Mentors**: Nicholas Sharp, Etienne Vouga, Josh Vekhter, Stephanie Wang

## Links
- **Papers:** [[Pinkall, 1993]](http://www.cs.jhu.edu/~misha/Fall09/Pinkall93.pdf)
- **Source Code**: [GitHub(MATLAB)](https://github.com/SGI-2021/nonmanifold-minimal-surfaces), [GitHub(C++)](https://github.com/evouga/SGI-nonmanifold)
- **Blog Posts**: [[Blog Post 1]](http://summergeometry.org/sgi2021/minimal-surfaces-but-periodic/), [[Blog Post 2]](http://summergeometry.org/sgi2021/minimal-surfaces-but-with-saddle-points/)

---
## Installation (Linux)
Clone from this repo

    git clone --recursive https://github.com/Zhecheng-Wang/nonmanifold-minimal-surfaces.git


Initialize build folder and compile the code

    mkdir build
    cd build
    cmake ..
    make

To run the program, run ``nonmanifold`` in the ``build`` folder.

### Installation Debug Notes
You can install ``SuiteSparse`` with

    sudo apt-get install libsuitesparse-dev

In case CMake cannot find ``METIS`` when running command ``cmake ..``, install ``METIS`` with

    sudo apt-get install libmetis-dev
