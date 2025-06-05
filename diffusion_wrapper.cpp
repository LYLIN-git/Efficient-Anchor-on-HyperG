// diffusion_wrapper.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h> 
#include "diffusion.h"      
#include <vector>
#include <iostream>
#include <iomanip> 

namespace py = pybind11;

Eigen::VectorXd py_top_level_diffusion( 
    const std::string& graph_path_str,
    const Eigen::VectorXd& seeds_vec_dense, 
    int T,
    double lam,
    double eps,          
    int schedule_val,   
    const std::string& label_txt_path_str
) {
    std::cerr << std::fixed << std::setprecision(10); std::cerr << "DEBUG: py_top_level_diffusion START" << std::endl;
    std::cerr << "DEBUG: [Wrapper VERY ENTRY] seeds_vec_dense.size(): " << seeds_vec_dense.size() << std::endl;
    if (seeds_vec_dense.size() > 0) {
        std::cerr << "DEBUG: [Wrapper VERY ENTRY] seeds_vec_dense.norm(): " << seeds_vec_dense.norm() << std::endl;
        std::cerr << "DEBUG: [Wrapper VERY ENTRY] seeds_vec_dense.head(10).transpose():\n" << seeds_vec_dense.head(std::min(10, (int)seeds_vec_dense.size())).transpose() << std::endl;
        bool found_first_nonzero = false; for (int i = 0; i < seeds_vec_dense.size(); ++i) { if (std::abs(seeds_vec_dense(i)) > 1e-12) { std::cerr << "DEBUG: [Wrapper VERY ENTRY] First non-zero in seeds_vec_dense at index " << i << " is " << seeds_vec_dense(i) << std::endl; found_first_nonzero = true; break; } }
        if (!found_first_nonzero) { std::cerr << "DEBUG: [Wrapper VERY ENTRY] No non-zero elements found in seeds_vec_dense (threshold 1e-12)." << std::endl; }
        if (seeds_vec_dense.size() >= 5) { std::cerr << "DEBUG: [Wrapper VERY ENTRY] seeds_vec_dense head(5): " << seeds_vec_dense.head(5).transpose() << std::endl;}
        else if (seeds_vec_dense.size() > 0) { std::cerr << "DEBUG: [Wrapper VERY ENTRY] seeds_vec_dense (all elements): " << seeds_vec_dense.transpose() << std::endl; }
        int nnz_seeds = 0; for(int i=0; i < seeds_vec_dense.size(); ++i) { if(std::abs(seeds_vec_dense(i)) > 1e-9) nnz_seeds++;} std::cerr << "DEBUG: [Wrapper VERY ENTRY] seeds_vec_dense non-zeros (approx > 1e-9): " << nnz_seeds << std::endl;
    } else { std::cerr << "DEBUG: [Wrapper VERY ENTRY] seeds_vec_dense is empty." << std::endl; }
    std::string labels_path_for_solver = label_txt_path_str; std::string preconditioner_name_for_solver = "degree"; int verbose_for_solver = 0; 
    GraphSolver solver(graph_path_str, labels_path_for_solver, preconditioner_name_for_solver, verbose_for_solver);
    if (solver.n == 0) { return Eigen::VectorXd(); } if (seeds_vec_dense.size() != solver.n) { return Eigen::VectorXd(solver.n).setZero(); }
    Eigen::VectorXd result_vector = solver.diffusion(seeds_vec_dense, T, lam, eps, schedule_val);
    std::cerr << "DEBUG: [Wrapper] Returned from solver.diffusion." << std::endl; 
    if (result_vector.size() > 0) {
         std::cerr << "DEBUG: [Wrapper] result_vector norm (AFTER solver.diffusion): " << result_vector.norm() << ", size: " << result_vector.size() << std::endl;
        if (result_vector.size() >= 5) { std::cerr << "DEBUG: [Wrapper] result_vector head(5): " << result_vector.head(5).transpose() << std::endl; }
        else if (result_vector.size() > 0) { std::cerr << "DEBUG: [Wrapper] result_vector (all elements): " << result_vector.transpose() << std::endl; }
        int nnz_result = 0; for(int i=0; i < result_vector.size(); ++i) if(std::abs(result_vector(i)) > 1e-9) nnz_result++;
        std::cerr << "DEBUG: [Wrapper] result_vector non-zeros (approx > 1e-9): " << nnz_result << std::endl;
    } else { std::cerr << "DEBUG: [Wrapper] result_vector from solver.diffusion is empty or size 0." << std::endl; }   
    std::cerr << "DEBUG: py_top_level_diffusion END" << std::endl;
    return result_vector;
}

PYBIND11_MODULE(diffusion, m) {
    m.doc() = "Python bindings for hypergraph diffusion"; 
    py::class_<GraphSolver>(m, "GraphSolver")
        .def(py::init<std::string, std::string, std::string, int>(),
             py::arg("graph_filename"), py::arg("label_filename"), py::arg("preconditioner"), py::arg("verbose")=0)
        .def(py::init<int, int, Eigen::VectorXd, std::vector<std::vector<int>>, int, std::vector<int>, int>(),
             py::arg("n_val"), py::arg("m_val"), py::arg("degree_val"), py::arg("hypergraph_val"),
             py::arg("label_count_val")=0, py::arg("labels_val")=std::vector<int>(), py::arg("verbose_val")=0)
        .def("infinity_subgradient", &GraphSolver::infinity_subgradient, py::arg("x_vec")) // Single argument
        .def("diffusion", &GraphSolver::diffusion, 
             py::arg("s_vec"), py::arg("T"), py::arg("lambda"), py::arg("h_step"), py::arg("schedule")=0)
        .def("compute_fx", &GraphSolver::compute_fx,
             py::arg("x_vec"), py::arg("s_vec"), py::arg("lambda"), py::arg("t_fx_param")=1)
        .def("compute_error", &GraphSolver::compute_error, py::arg("x_vec"))
        .def("run_diffusions", &GraphSolver::run_diffusions,
            py::arg("graph_name_param"), py::arg("repeats_param"), py::arg("T_param"),
            py::arg("lambda_param"), py::arg("h_param"), py::arg("minimum_revealed"),
            py::arg("step_param"), py::arg("maximum_revealed"), py::arg("schedule_param")=0)
        .def_readwrite("n", &GraphSolver::n)
        .def_readwrite("m", &GraphSolver::m)
        .def_readwrite("graph_name", &GraphSolver::graph_name)
        .def_readwrite("degree", &GraphSolver::degree)
        .def_readwrite("hypergraph", &GraphSolver::hypergraph)
        .def_readwrite("weights", &GraphSolver::weights)
        .def_readwrite("hypergraph_node_weights", &GraphSolver::hypergraph_node_weights)
        .def_readwrite("center_id", &GraphSolver::center_id)
        .def_readwrite("label_count", &GraphSolver::label_count)
        .def_readwrite("labels", &GraphSolver::labels)
        .def_readwrite("early_stopping", &GraphSolver::early_stopping)
        .def_readwrite("verbose", &GraphSolver::verbose)
        .def_readwrite("has_hyperedge_centers", &GraphSolver::has_hyperedge_centers)
        .def_readwrite("preconditionerType", &GraphSolver::preconditionerType);
    m.def("diffusion", &py_top_level_diffusion,
          "Top-level wrapper for hypergraph diffusion via GraphSolver.",
          py::arg("graph_path_str"), py::arg("seeds_vec_dense"), py::arg("T"),
          py::arg("lam"), py::arg("eps"), py::arg("schedule_val"), py::arg("label_txt_path_str"));
}