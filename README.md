# Systems Optimization Research
Repo with code of the algorithms developed of predictive control systems, GPC (Generalized Predictive Control) and IHMPC (Infinite Horizon Model Predictive Control). Internship report and base article used for the implementations.

## Repo structure:
- ./requirements.txt : python dependencies to execute the algorithms
- ./doc : articles and reports
- ./img : pictures and screenshots from the experiments
- ./src : source code for the algorithms implementation
  - /gpc : GPC algorithm files
  - /ihmpc : IHMPC algorithm files
  - /ethylene_oxide_gpc.py : script with simulation with the application of GPC algorithm to control an ethylene oxide plant
  - /ethylene_oxide_ihmpc.py : script with simulation with the application of IHMPC algorithm to control an ethylene oxide plant

### Prerequisites
For the execution of the IHMPC algorithm, it's used the solver [OSQP](http://osqp.readthedocs.io/en/latest/installation/python.html). The following dependencies must be installed on your machine:
- [GCC](https://gcc.gnu.org/)
- [CMake](https://cmake.org/)

  ```
  curl https://cmake.org/files/v3.10/cmake-3.10.3-Linux-x86_64.sh -o /tmp/curl-install.sh \
        && chmod u+x /tmp/curl-install.sh \
        && mkdir /usr/bin/cmake \
        && /tmp/curl-install.sh --skip-license --prefix=/usr/bin/cmake \
        && rm /tmp/curl-install.sh
  ```

## Instructions to run:
- `git clone git@github.com:igoryamamoto/internship-code.git`
- `cd internship-code`
- `pip3 install -r requirements.txt`
- `pip3 install osqp`
- `cd src`
- `python3 ethylene_oxide_gpc.py`
- `python3 ethylene_oxide_ihmpc.py`
