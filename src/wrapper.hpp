#ifndef ABC_HPP
#define ABC_HPP

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <string>

class ABC
{
  // Other declarations 
public:
    ABC(const std::string& someConfigFile);
    PyObject* doSomething(PyObject* image, PyObject* particles, PyObject* features, PyObject* out);
    PyObject* test(PyObject* image, PyObject* particle); // We want our python code to be able to call this function to do some processing using OpenCV and return the result.
  // Other declarations
};

#endif
