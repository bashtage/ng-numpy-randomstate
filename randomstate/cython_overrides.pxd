cdef extern from "Python.h":
    double PyFloat_AsDouble(object ob)
    long PyInt_AsLong(object ob)
    int PyErr_Occurred()
    void PyErr_Clear()
