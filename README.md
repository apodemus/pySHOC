
pyshoc
======

**pyshoc** is a suite of object oriented tools for analysing data from the
Sutherland High Speed Optical Cameras (SHOC). 📷🔭

## Installation
```shell
pip install pyshoc
```

## Use
To run the reduction pipeline on a set of fits files located in a terget diretory
(multiple observations of a single target, possibly including calibration files):
```shell
pyshoc /path/to/data/folder -o /path/to/output/products -tel 74 --target "CTCV J1928-5001"
```
<!-- or equivalently
```shell
> python shoc/pipeline /path/to/data
``` -->

## Documentation
* We are working on official documentation. 
* For the moment the best place to see how to use pyshoc is the 
[example notebook](https://nbviewer.jupyter.org/github/astromancer/pyshoc/blob/master/pyshoc/example/pyshoc.demo.ipynb)


## Contributing
 Pull requests are welcome!
1. Fork it (<https://github.com/astromancer/pyshoc/fork>)
2. Create your feature branch (`git checkout -b feature/rad`)
3. Commit your changes (`git commit -am 'Add some cool feature 😎'`)
4. Push to the branch (`git push origin feature/rad`)
5. Create a new Pull Request


## Contact
* e-mail: hannes@saao.ac.za

<!-- ### Third party libraries
* see [LIBRARIES](https://github.com/username/sw-name/blob/master/LIBRARIES.md) files -->

## License 
* see [LICENSE](https://github.com/astromancer/pyshoc/blob/master/LICENSE)

## Version 
* 1.2.0
