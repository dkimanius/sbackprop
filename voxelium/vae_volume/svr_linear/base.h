#pragma once
#ifndef SVR_LINEAR_BASE_H
#define SVR_LINEAR_BASE_H

#include <torch/script.h>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>

#define SPECTRAL_WEIGHT_EPS 1e-6

#define CHECK_SIZE_DIM0(x, SIZE) TORCH_CHECK(x.size(0) == SIZE, \
    #x ".size(0) (", x.size(0), ") != " #SIZE " (", SIZE, "). In ", __FILE__, ":", __LINE__)
#define CHECK_SIZE_DIM1(x, SIZE) TORCH_CHECK(x.size(1) == SIZE, \
    #x ".size(1) (", x.size(1), ") != " #SIZE " (", SIZE, "). In ", __FILE__, ":", __LINE__)
#define CHECK_SIZE_DIM2(x, SIZE) TORCH_CHECK(x.size(2) == SIZE, \
    #x ".size(2) (", x.size(2), ") != " #SIZE " (", SIZE, "). In ", __FILE__, ":", __LINE__)
#define CHECK_SIZE_DIM3(x, SIZE) TORCH_CHECK(x.size(3) == SIZE, \
    #x ".size(3) (", x.size(3), ") != " #SIZE " (", SIZE, "). In ", __FILE__, ":", __LINE__)

#define CHECK_DTYPE(x, DTYPE) TORCH_CHECK(x.dtype() == DTYPE, \
    #x " has the wrong data type (", x.dtype(), "), expecting " \
    #DTYPE, " (", DTYPE, "). In ", __FILE__, ":", __LINE__)

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), \
    #x " must be a CPU tensor. In ", __FILE__, ":", __LINE__)

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), \
    #x " must be contiguous. In ", __FILE__, ":", __LINE__)

#define CHECK_DIM(x, DIM) TORCH_CHECK(x.dim() == DIM, \
    #x " has wrong number of dimensions. In ", __FILE__, ":", __LINE__)

#define CHECK_CPU_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

/** Linear interpolation
 *
 * From low (when a=0) to high (when a=1). The following value is returned
 * (equal to (a*h)+((1-a)*l)
 */
#ifndef LIN_INTERP
    #define LIN_INTERP(a, l, h) ((l) + ((h) - (l)) * (a))
#endif

template<std::size_t max>
struct dispatch_bools
{
    template<std::size_t N, class F, class...Bools>
    void operator()( std::array<bool, N> const& input, F&& continuation, Bools... )
    {
        if (input[max-1])
            dispatch_bools<max-1>{}( input, continuation, std::integral_constant<bool, true>{}, Bools{}... );
        else
            dispatch_bools<max-1>{}( input, continuation, std::integral_constant<bool, false>{}, Bools{}... );
    }
};

template<>
struct dispatch_bools<0>
{
    template<std::size_t N, class F, class...Bools>
    void operator()( std::array<bool, N> const& input, F&& continuation, Bools... )
    {
        continuation( Bools{}... );
    }
};

#endif // SVR_LINEAR_BASE_H