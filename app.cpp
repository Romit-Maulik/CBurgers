// cmake .. // From build
// make
// export PATH=/home/rmlans/anaconda3/envs/tf2_env/bin:$PATH
// export LD_LIBRARY_PATH=/home/rmlans/anaconda3/envs/tf2_env/lib:$LD_LIBRARY_PATH

#include <iostream>
#include <time.h>
#include <math.h>
#include <string.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <thread>
#include <chrono>         // std::chrono::seconds

struct solution_info
{
  double *u;
  double *u_temp;
  double t;
  double DT;
  double ft;
  PyObject *pcollection_func;
};

double PI = 3.1415926535;
double NU = 0.01;
int NX = 256;
double DT = 0.001;
double FT_1 = 1.0;
double FT_2 = 2.0;

void collect_data(PyObject* pcollection_func, double* u);
void save_data(PyObject* pdata_save_func);
void analyse_data(PyObject* panalyses_func);
void initialize(double* u);
void update_solution(double* u, double* u_temp);
void fdm_solve(solution_info* current_sol);


int main(int argc, char *argv[])
{

    // Some python initialization
    int some_int = 0;
    int num_steps = 0;


    Py_Initialize();
    PyGILState_STATE gil_state = PyGILState_Ensure();
    
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");

    std::cout << "Initializing numpy library" << std::endl;
    // initialize numpy array library
    import_array1(-1);
    
    std::cout << "Loading python module" << std::endl;
    PyObject *pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName); // finished with this string so release reference
    std::cout << "Loaded python module" << std::endl; 

    std::cout << "Loading functions from module" << std::endl;
    PyObject *pcollection_func = PyObject_GetAttrString(pModule, "collection_func");
    PyObject *pdata_save_func = PyObject_GetAttrString(pModule, "data_save_func");
    PyObject *panalyses_func = PyObject_GetAttrString(pModule, "analyses_func");
    Py_DECREF(pModule); // finished with this string so release reference
    std::cout << "Loaded functions" << std::endl;

    PyGILState_Release(gil_state);

    // Do the array initialization business for the solution field
    double u[NX+2]; // 2 GHOST POINTS
    initialize(u);

    std::cout << u << std::endl;

    double u_temp[NX+2]; // 2 GHOST POINTS
    initialize(u_temp);

    // Time loop for evolution of the Burgers equation
    clock_t start, end;
    double t, cpu_time_used;

    // FDM solver
    start = clock();
    gil_state = PyGILState_Ensure();
    std::cout<<"Thread for main PDE solve with thread id " << std::this_thread::get_id() << std::endl;
    t = 0.0;  
    do{
      // PDE update

      update_solution(u,u_temp);

      // Exchanging data with python
      collect_data(pcollection_func,u);

      // std::cout << "time = " << t << std::endl;;
      t = t + DT;

    }while(t<FT_1);
    PyGILState_Release(gil_state);
    

    // Save the solution in python in temp array that we use for analysis
    std::cout<<"Saving data in a temp array in Python" << std::endl;
    save_data(pdata_save_func);
    // Wait for the copy
    std::this_thread::sleep_for (std::chrono::seconds(3));

    // At this point - we want to switch to task parallelism
    // Create struct to send it across to different thread

    std::cout<<"Starting task parallelism" << std::endl;
    
    solution_info current_sol;
    current_sol.u = u;
    current_sol.u_temp = u_temp;
    current_sol.t = t;
    current_sol.DT = DT;
    current_sol.ft = FT_2;
    current_sol.pcollection_func = pcollection_func;

    std::thread thread1 (fdm_solve,&current_sol);

    // Analyse data in main thread
    analyse_data(panalyses_func);

    // Wait for FDM solve to finish
    thread1.join();

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    std::cout<<"CPU time = "<< cpu_time_used << std::endl;

    return 0;
}

void fdm_solve(solution_info* current_sol)
{
    
    std::cout<<"Thread for parallel PDE solve with thread id " << std::this_thread::get_id() << std::endl;

    do{
      // PDE update

      update_solution(current_sol->u,current_sol->u_temp);

      // // Exchanging data with python
      PyGILState_STATE gil_state = PyGILState_Ensure();
      collect_data(current_sol->pcollection_func,current_sol->u);
      PyGILState_Release(gil_state);

      std::cout << "time = " << current_sol->t << std::endl;
      current_sol->t = current_sol->t + current_sol->DT;

    }while(current_sol->t<current_sol->ft);
}

void initialize(double* u)
{
  double x;

  for (int i = 1; i < NX+1; i++)
  {
      x = (double) (i-1)/(NX) * 2.0 * PI;
      u[i] = sin(x);
  }

  // Update periodic BCs
  u[0] = u[NX];
  u[NX+1] = u[1];
}

void update_solution(double* u, double* u_temp)
{
  double dx = 2.0 * PI/NX;

  // loop over the array, updating solution u with a finite difference method
  // (based on values at the previous time in the neighborhood)
  // Burgers' equation: u_t + u*u_x = nu * u_xx
  // skips updating ghost points, one on either end

  for (int i = 1; i < NX+1; i++)
  {
      u[i] = u_temp[i] + NU*DT/(dx*dx)*(u_temp[i+1]+u_temp[i-1]-2.0*u_temp[i]) - DT/(2*dx)*(u_temp[i+1]-u_temp[i-1])*u_temp[i];
  }

  for (int i = 1; i < NX+1; i++)
  {
      u_temp[i] = u[i];
  }  

  // Update periodic BCs
  u[0] = u[NX];
  u[NX+1] = u[1];

  // Update periodic BCs
  u_temp[0] = u_temp[NX];
  u_temp[NX+1] = u_temp[1];

}


void save_data(PyObject* pdata_save_func)
{
  PyGILState_STATE gil_state = PyGILState_Ensure();
  PyArrayObject *pValue = (PyArrayObject*)PyObject_CallObject(pdata_save_func, nullptr); //Casting to PyArrayObject
  // std::cout << "Called python data collection function successfully"<<std::endl;
  Py_DECREF(pValue);
  PyGILState_Release(gil_state);
}


void collect_data(PyObject* pcollection_func, double* u)
{

  PyObject *pArgs = PyTuple_New(1);
  
  //Numpy array dimensions
  npy_intp dim[] = {NX+2};
  // create a new array
  PyObject *array_1d = PyArray_SimpleNewFromData(1, dim, NPY_FLOAT64, u);
  PyTuple_SetItem(pArgs, 0, array_1d);
  PyArrayObject *pValue = (PyArrayObject*)PyObject_CallObject(pcollection_func, pArgs); //Casting to PyArrayObject
  // std::cout << "Called python data collection function successfully"<<std::endl;

  Py_DECREF(pArgs);
  Py_DECREF(pValue);
  // We don't need to decref array_1d because PyTuple_SetItem steals a reference
}

void analyse_data(PyObject* panalyses_func)
{ 
  std::cout<<"Thread for python analysis function with id " << std::this_thread::get_id() << std::endl;
  PyGILState_STATE gil_state = PyGILState_Ensure();

  // // panalsyses_func doesn't require an argument so pass nullptr 
  PyArrayObject* pValue = (PyArrayObject*)PyObject_CallObject(panalyses_func, nullptr);  
  std::cout << "Called python analyses function successfully"<<std::endl;

  PyGILState_Release(gil_state);

  // Printing out values of the SVD eigenvectors of the first and second modes for each field DOF
  for (int i = 0; i < 10; ++i) 
  {
    double* current = (double*) PyArray_GETPTR2(pValue, 0, i); // row 0, column i
    std::cout << "First mode value: " << *current << std::endl;
  }

  for (int i = 0; i < 10; ++i)
  {
    double* current = (double*) PyArray_GETPTR2(pValue, 1, i); // row 1, column i
    std::cout << "Second mode value: " << *current << std::endl;
  }

  Py_DECREF(pValue);

}
