.. _gemm:

GEMM
====

``onHost::gemm`` computes

.. math::

   C := \alpha \cdot \operatorname{op}(A) \cdot \operatorname{op}(B) + \beta \cdot C

with :math:`A` logically :math:`M \times K`, :math:`B` logically :math:`K \times N` and :math:`C` logically
:math:`M \times N`.
The operation is enqueued on an *alpaka* queue and follows its asynchronous semantics.
The caller keeps the view storage alive until the queue has finished the operation.

API
---

The call takes the queue first.
The executor is optional.
Without an executor, or with ``alpaka::exec::anyExecutor``, the default executor of the queue's device is used.

.. code-block:: cpp

   void onHost::gemm(
       Queue auto const& queue,
       [Executor auto const& exec,]
       GemmView<T_C> c,
       GemmView<T_A> a,
       GemmView<T_B> b,
       T_Alpha alpha,
       T_Beta beta,
       GemmNumericMode mode = GemmNumericMode::fp32);

``GemmView<T>`` is a non-owning view over existing storage.
It stores the data pointer, the logical rows and columns, the leading dimension, a ``GemmLayout``
(``rowMajor`` or ``colMajor``) and a ``GemmOp`` (``none`` or ``transpose``).
The layout, transpose flag and leading dimension belong to the view, so a padded leading dimension or a shifted
subview needs no separate argument.
The output view must use ``GemmOp::none``.

.. literalinclude:: ../../snippets/example/250_gemm.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-gemmCall
    :end-before: END-TUTORIAL-gemmCall
    :dedent:

Numerical modes
---------------

The numerical mode is an explicit part of the contract and must match the view value types:

``GemmNumericMode::fp32``
    ``float`` storage for ``C``, ``A`` and ``B``, FP32 accumulation.

``GemmNumericMode::fp64``
    ``double`` storage for ``C``, ``A`` and ``B``, FP64 accumulation.

``GemmNumericMode::fp16ToFp32``
    ``alpaka::Float16`` storage for ``A`` and ``B``, ``float`` storage for ``C``.
    The inputs are converted exactly to FP32, products and accumulation use FP32, and the result is stored as FP32.

``GemmNumericMode::bf16ToFp32``
    ``alpaka::BFloat16`` storage for ``A`` and ``B`` with the same FP32 product, accumulation and output contract.

A mode and value-type mismatch is rejected before launch.
The two 16-bit storage types are not interchangeable.

Layouts and transposes
----------------------

Row-major and column-major storage, padded leading dimensions, shifted subviews and the four NN/NT/TN/TT transpose
combinations are supported.
Dimensions, leading dimensions, overlapping output and input storage, null pointers of views that are read or
written, and overflow of size or launch arithmetic are validated on the host before work is enqueued.
Invalid user input throws ``std::invalid_argument``.

Precision contract
------------------

The accumulation type is selected by the mode.
The reference tolerance model lives in ``validation/numerics/tolerance.hpp`` and the kernel tests compare against an
independent higher-precision reference on each supported layout and transpose combination.
FP16 and BF16 inputs contribute their exact FP32 value to the FP32 accumulation.
The value stays exact also inside the FP32 subnormal range: BF16 subnormal inputs widen to FP32 subnormals and
contribute as finite FP32 operands.
Host builds with the Intel C++ compilers (``icx``, ``icpx``) therefore pin the host FP32 semantics to
``-ffp-model=precise``: the default host FP model of these compilers flushes FP32 denormals to zero and would
break the contract.
Setting ``alpaka_FAST_MATH`` or ``alpaka_FTZ`` to ``ON`` opts out of the pin and waives this guarantee.

Plan diagnostics
----------------

The current implementation exposes no plan object, expert override or tuning-policy diagnostics.
Tile size, K step and block thread count are derived from the compiled SIMD and executor traits and from the
problem shape.
A problem smaller than four rows or columns keeps the 4x4 fallback tile.
