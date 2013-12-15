PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# location of the Boost Python include files and library
BOOST_INC = /usr/local/include/boost
BOOST_LIB = /usr/local/lib

OPENCV_LIB = `pkg-config --libs opencv`
OPENCV_CFLAGS = `pkg-config --cflags opencv`


TARGET = pysomemodule
SRC = pysomemodule.cpp abc.cpp
OBJ = pysomemodule.o abc.o

$(TARGET).so: $(OBJ)
	g++ -O3 -march=native -shared $(OBJ) -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so $(OPENCV_LIB) 

$(OBJ): $(SRC)
	g++ -O3 -march=native -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(OPENCV_CFLAGS) -fPIC -c $(SRC)

clean:
	rm -f $(OBJ)
	rm -f $(TARGET).so
