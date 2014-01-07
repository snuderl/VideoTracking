#include <string>    
#include<boost/python.hpp>
#include "abc.hpp"

using namespace boost::python;

BOOST_PYTHON_MODULE(pysomemodule)
{
    class_<ABC>("ABC", init<const std::string &>())
      .def(init<const std::string &>())
      .def("doSomething", &ABC::doSomething)
      .def("test", &ABC::test)  // doSomething is the method in class ABC you wish to expose. One line for each method (or function depending on how you structure your code). Note: You don't have to expose everything in the library, just the ones you wish to make available to python.
    ;
}
