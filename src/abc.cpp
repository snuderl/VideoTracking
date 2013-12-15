

#include "abc.hpp"

#include "numpy/ndarrayobject.h"
#include "opencv2/core/core.hpp"
#include <cv.h>
#include <iostream>



// The following conversion functions are taken from OpenCV's cv2.cpp file inside modules/python/src2 folder.
static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

using namespace cv;

static PyObject* failmsgp(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() {}
    ~NumpyAllocator() {}

    void allocate(int dims, const int* sizes, int type, int*& refcount,
                  uchar*& datastart, uchar*& data, size_t* step)
    {
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                      depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                      depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                      depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i;
        npy_intp _sizes[CV_MAX_DIM+1];
        for( i = 0; i < dims; i++ )
        {
            _sizes[i] = sizes[i];
        }

        if( cn > 1 )
        {
            /*if( _sizes[dims-1] == 1 )
                _sizes[dims-1] = cn;
            else*/
                _sizes[dims++] = cn;
        }

        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);

        if(!o)
        {
            CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        }
        refcount = refcountFromPyObject(o);

        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*)o);
        for( i = 0; i < dims - (cn > 1); i++ )
            step[i] = (size_t)_strides[i];
        datastart = data = (uchar*)PyArray_DATA((PyArrayObject*)o);
    }

    void deallocate(int* refcount, uchar*, uchar*)
    {
        PyEnsureGIL gil;
        if( !refcount )
            return;
        PyObject* o = pyObjectFromRefcount(refcount);
        Py_INCREF(o);
        Py_DECREF(o);
    }
};

NumpyAllocator g_numpyAllocator;

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

static int pyopencv_to(const PyObject* o, Mat& m, const char* name = "<unknown>", bool allowND=true)
{
    //NumpyAllocator g_numpyAllocator;
    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
        return true;
    }

    if( !PyArray_Check(o) )
    {
        failmsg("%s is not a numpy array", name);
        return false;
    }

    int typenum = PyArray_TYPE((PyArrayObject*)o);
    int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U : typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        failmsg("%s data type = %d is not supported", name, typenum);
        return false;
    }

    int ndims = PyArray_NDIM((PyArrayObject*)o);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("%s dimensionality (=%d) is too high", name, ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS((PyArrayObject*)o);
    const npy_intp* _strides = PyArray_STRIDES((PyArrayObject*)o);
    bool transposed = false;

    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    if( ndims == 0 || step[ndims-1] > elemsize ) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if( ndims >= 2 && step[0] < step[1] )
    {
        std::swap(size[0], size[1]);
        std::swap(step[0], step[1]);
        transposed = true;
    }

    if( ndims == 3 && size[2] <= CV_CN_MAX && step[1] == elemsize*size[2] )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if( ndims > 2 && !allowND )
    {
        failmsg("%s has more than 2 dimensions", name);
        return false;
    }

    m = Mat(ndims, size, type, PyArray_DATA((PyArrayObject*)o), step);

    if( m.data )
    {
        m.refcount = refcountFromPyObject(o);
        m.addref(); // protect the original numpy array from deallocation
                    // (since Mat destructor will decrement the reference counter)
    };
    m.allocator = &g_numpyAllocator;

    if( transposed )
    {
        Mat tmp;
        tmp.allocator = &g_numpyAllocator;
        transpose(m, tmp);
        m = tmp;
    }
    return true;
}

static PyObject* pyopencv_from(const Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->refcount || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        m.copyTo(temp);
        p = &temp;
    }
    p->addref();
    return pyObjectFromRefcount(p->refcount);
}

// Note the import_array() from NumPy must be called else you will experience segmentation faults.
ABC::ABC(const std::string &someConfigFile)
{
  // Initialization code. Possibly store someConfigFile etc.
  import_array(); // This is a function from NumPy that MUST be called.
  // Do other stuff
}
#include <opencv2/highgui/highgui.hpp>
cv::Mat process(cv::Mat image, cv::Mat particles, cv::Mat features, cv::Mat indices){
	cv::Mat out(particles.rows, features.rows*3, CV_64F, double(0.0));


	for(int i = 0; i < particles.rows; i++){
		float x1 = particles.at<float>(i,0);
		float y1 = particles.at<float>(i,1);
		float width = particles.at<float>(i,2);
		float height = particles.at<float>(i,3);

		cv::Mat subrect;
		cv::Mat integral;

		cv::getRectSubPix(image, cv::Size(int(width), int(height)), cv::Point2f(x1+width/2,y1+height/2), subrect);


		cv::integral(subrect, integral, CV_64F);
		cv::Mat rgb[3];
		cv::split(integral, rgb);

		int h = integral.rows-1;
		int w = integral.cols-1;

	

		for(int l = 0; l < indices.rows; l++){
			int indice = indices.at<int>(l,0);
			int k = indice/3;
			int color = indice % 3;


			int x1 = static_cast<int>(features.at<float>(k,0)*h);
			int y1 = static_cast<int>(features.at<float>(k,1)*w);
			int x2 = static_cast<int>(features.at<float>(k,2)*h);
			int y2 = static_cast<int>(features.at<float>(k,3)*w);

			double A,B,C,D;
			//cout << x1 << " " << y1 << endl;
			A = rgb[color].at<double>(x1,y1);
			D = rgb[color].at<double>(x2,y2);
			C = rgb[color].at<double>(x1,y2);
			B = rgb[color].at<double>(x2,y1);



			double outer = A + D - C - B;
	
			//std::cout << "Coords" << x1 << " "<< y1 << " " << x2 << " " << y2 << std::endl;
			x1 = static_cast<int>(features.at<float>(k,4)*h);
			y1 = static_cast<int>(features.at<float>(k,5)*w);
			x2 = static_cast<int>(features.at<float>(k,6)*h);
			y2 = static_cast<int>(features.at<float>(k,7)*w);
			A = rgb[color].at<double>(x1,y1);
			D = rgb[color].at<double>(x2,y2);
			C = rgb[color].at<double>(x1,y2);
			B = rgb[color].at<double>(x2,y1);
		
			
			double inner = A + D - C - B;
			double sum = outer - (2 *inner);


			
			if(sum < 0){
					
	
	 			//std::cout << "shape " << h+1 << " "<< w+1 <<  std::endl;
	 			//std::cout << "coords" << x1 << " "<< y1 << " " << x2 << " " << y2 << std::endl;
				//std::cout << A << " "<< D << " " << C << " " << B << std::endl;

				//x1 = static_cast<int>(features.at<float>(k,0)*h);
				//y1 = static_cast<int>(features.at<float>(k,1)*w);
				//x2 = static_cast<int>(features.at<float>(k,2)*h);
				//y2 = static_cast<int>(features.at<float>(k,3)*w);
				//A = rgb[color].at<double>(x1,y1);
				//D = rgb[color].at<double>(x2,y2);
				//C = rgb[color].at<double>(x1,y2);
				//B = rgb[color].at<double>(x2,y1);

	 			//std::cout << x1 << " "<< y1 << " " << x2 << " " << y2 << std::endl;
				//std::cout << A << " "<< D << " " << C << " " << B << std::endl;

				//std::cout << sum << " " << outer << " " << inner << std::endl;      
				sum = 0;
			}

			out.at<double>(i, indice) = sum;
		}
		
	}

	return out;
}


// The conversions functions above are taken from OpenCV. The following function is 
// what we define to access the C++ code we are interested in.
PyObject* ABC::doSomething(PyObject* image, PyObject* particles, PyObject* features, PyObject* indices)
{
    cv::Mat cvImage;
    cv::Mat cvParticles;
    cv::Mat cvFeatures;
    cv::Mat cvIndices;
    pyopencv_to(image, cvImage); // From OpenCV's source
    pyopencv_to(particles, cvParticles); // From OpenCV's source
    pyopencv_to(features, cvFeatures); // From OpenCV's source
    pyopencv_to(indices, cvIndices); // From OpenCV's source
   
    cv::Mat cvOut = process(cvImage, cvParticles, cvFeatures, cvIndices);

    return pyopencv_from(cvOut); // From OpenCV's source
}

PyObject* ABC::test(PyObject* image, PyObject* particle)
{
    cv::Mat cvImage, cvParticle, rect;
    pyopencv_to(image, cvImage); 
    pyopencv_to(particle, cvParticle); 

	float x1 = cvParticle.at<float>(0,0);
	float y1 = cvParticle.at<float>(0,1);
	float width = cvParticle.at<float>(0,2);
	float height = cvParticle.at<float>(0,3);

   cv::getRectSubPix(cvImage, cv::Size(int(width), int(height)), cv::Point2f(x1+width/2,y1+height/2), rect);

	std::cout << rect.rows << std::endl;
   
    return pyopencv_from(rect); // From OpenCV's source
}


