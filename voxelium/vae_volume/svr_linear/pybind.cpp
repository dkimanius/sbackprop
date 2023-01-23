#include "vae_volume/svr_linear/trilinear_projection.h"
#include "vae_volume/svr_linear/volume_extraction.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "trilinear_projection_forward",
        &trilinear_projection_forward,
        "Trilinear projector forward",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("rot_matrix"),
        py::arg("grid2d_coord"),
        py::arg("grid3d_index"),
        py::arg("max_r")
    );

    m.def(
        "trilinear_projection_backward",
        &trilinear_projection_backward,
        "Trilinear projector backward",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("rot_matrix"),
        py::arg("input_spectral_weight"),
        py::arg("grid2d_grad"),
        py::arg("grid2d_coord"),
        py::arg("grid3d_index"),
        py::arg("sparse_grad"),
        py::arg("max_r")
    );

    m.def(
        "volume_extraction_forward",
        &volume_extraction_forward,
        "Volume extraction forward",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("grid3d_index"),
        py::arg("max_r")
    );

    m.def(
        "volume_extraction_backward",
        &volume_extraction_backward,
        "Volume extraction backward",
        py::arg("input_spectral_weight"),
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("grad_output"),
        py::arg("grid3d_index")
    );
}
