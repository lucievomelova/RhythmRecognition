# RhythmRecognition
A python package for rhythm recognition in music.

## Installation and Usage
The package is located in `RhythmRecognition/` directory. It was developed in Python 3.11.9.

Before using the package, you need to install other necessary python modules used in the package. If you want to 
use just the provided package, without the provided examples, run:
```bash
python -m pip install -r .\RhythmRecognition\requirements.txt 
```
This will install only the requirements needed in the package. 

If you want to install packages used in the examples and tests as well, run:
```bash
python -m pip install -r .\requirements.txt
```


## Documentation and examples
There is documentation provided for the whole package. It was generated using `pdoc` 
which is an automatic documentation generator. It can be found in the `documentation/` directory.
There are also jupyter notebooks showing example usages of the provided code, they are located in the `exmaples/` 
directory. 

## Tests
Code for testing can be found in the `tests/` directory. However, the songs that the tests were conducted on are 
not provided, as the total size would be too big. The only song provided wwith this project
can be found in `examples/audio_files`. It is Spark by Vexento. 


