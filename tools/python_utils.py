#include <Python.h>
#include <iostream>
#include <string>
#include <cstdlib>

void InitializePythonWithVenv(const char* venv_path) {
    std::string python_home = std::string(venv_path) + "/";

    // Set PYTHONHOME to point to the virtual environment
    setenv("PYTHONHOME", python_home.c_str(), 1);

    // Set PYTHONPATH to include site-packages
    std::string python_path = python_home + "lib/pythonX.Y/site-packages";
    setenv("PYTHONPATH", python_path.c_str(), 1);

    // Initialize Python
    Py_Initialize();
}

void CallPythonFunction(const std::string& script_path,
                        const std::string& module_name,
                        const std::string& function_name,
                        const std::string& arg1,
                        int arg2) {
    // Initialize Python interpreter
    Py_Initialize();

    // Add script directory to sys.path
    std::string directory = script_path.substr(0, script_path.find_last_of("/\\"));
    PyObject* sys_path = PySys_GetObject("path");
    PyObject* path = PyUnicode_FromString(directory.c_str());
    PyList_Append(sys_path, path);
    Py_DECREF(path);

    // Import Python module
    PyObject* pModule = PyImport_ImportModule(module_name.c_str());
    if (!pModule) {
        PyErr_Print();
        std::cerr << "Failed to load Python module: " << module_name << std::endl;
        Py_Finalize();
        return;
    }

    // Get function from module
    PyObject* pFunc = PyObject_GetAttrString(pModule, function_name.c_str());
    if (!pFunc || !PyCallable_Check(pFunc)) {
        PyErr_Print();
        std::cerr << "Function " << function_name << " not found or not callable." << std::endl;
        Py_DECREF(pModule);
        Py_Finalize();
        return;
    }

    // Prepare arguments (Python tuple)
    PyObject* pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(arg1.c_str())); // String argument
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(arg2)); // Integer argument

    // Call the Python function
    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    if (pResult) {
        // Convert result to C++ string
        std::cout << "Python returned: " << PyUnicode_AsUTF8(pResult) << std::endl;
        Py_DECREF(pResult);
    } else {
        PyErr_Print();
    }

    // Cleanup
    Py_DECREF(pArgs);
    Py_DECREF(pFunc);
    Py_DECREF(pModule);
    Py_Finalize();
}

int main() {
    std::string script_path = "/path/to/my_script.py"; // Adjust path
    std::string module_name = "my_script"; // Module name without ".py"
    std::string function_name = "greet";

    CallPythonFunction(script_path, module_name, function_name, "Alice", 25);

    return 0;
}
