# Estágio - Ígor Yamamoto
Repositório com os códigos dos algoritmos de controle preditivo (GPC, IHMPC), relatório de estágio e artigos-base para a implementação.

## Estrutura do repositório:
- ./requirements.txt : arquivo com as bibliotecas do python necessárias para executar os arquivos
- ./doc : diretório com documentos (relatório de estágio, artigos-base)
- ./img : diretório contendo imagens de experimentos
- ./src : diretório contendo o código-fonte em python dos algoritmos
  - /gpc : arquivos com a implementação do algoritmo GPC
  - /ihmpc : arquivo com a implementação do algoritmo IHMPC
  - /ethylene_oxide_gpc.py : script com experimento de controle da planta de óxido de etileno com GPC
  - /ethylene_oxide_ihmpc.py : script com experimento de controle da planta de óxido de etileno com IHMPC
  
## Instruções para execução dos experimentos:
- Utilizar Python 3
- `git clone git@github.com:igoryamamoto/internship-code.git`
- `cd internship-code`
- `pip3 install -r requirements.txt`
- `cd src`
- `python3 ethylene_oxide_gpc.py`
- `python3 ethylene_oxide_ihmpc.py`

### Pré-requisitos de instalação:
Para a execução do algoritmo IHMPC, o solver [OSQP](http://osqp.readthedocs.io/en/latest/installation/python.html) é utilizado. As seguintes dependências devem ser instaladas na máquina:
- [GCC](https://gcc.gnu.org/)
- [CMake](https://cmake.org/)
