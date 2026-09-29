Lockstep and Portable SIMD
===========================

Lockstep lets you write plain per-element or loop code once and have *alpaka* map it onto
the SIMD units of the selected backend.
On a CPU the backend may use wide vector registers, on a GPU the same code runs as SIMT
lanes of a warp; you do not write intrinsics and you do not write datatype-specific code.
The same kernel source therefore stays portable and can still use the full width of the hardware.

Motivation
----------

- **Write scalar-looking code, run it in parallel.** The functor you pass to a lockstep scope is written as if it
  processed a single element. *alpaka* decides the SIMD width for the backend and partitions the logical iteration
  space over the workers of a work group.
- **Portable performance.** The backend chooses the SIMD width per datatype and device (for example 4 ``int32_t``
  per CPU vector register) so you do not have to provide hand-written fallbacks.
- **Per-worker distributed registers.** ``scope.var<T>()`` gives every worker a small private array in registers or
  local memory. The array is automatically partitioned over the group, which makes intermediate values such as a
  tile, a shifted copy, or a per-element result cheap to keep in-kernel.
- **No data-size restrictions.** The iteration space does not need to be a multiple of the SIMD width; *alpaka*
  handles the remainder (tail) iterations.

A Basic Lockstep Kernel With a Distributed Array
------------------------------------------------

``onAcc::makeLockstep(acc, workGroup, logicalExtent)`` creates a scope over a **compile-time known** logical
iteration space and distributes it over the workers of ``workGroup``.
The scope has two important members:

- ``scope.var<T>()`` returns a distributed per-worker array of ``T``. Its size is derived from the logical extent and
  the worker count, so each worker only allocates the slots it actually owns.
- ``scope.concurrent(fn, args...)`` executes ``fn`` once per lockstep iteration. The first argument is the lockstep
  index; the remaining arguments are the bound data arguments. Inside ``fn`` a handle from ``var()`` is exposed as a
  SIMD reference: assign to it to store and call ``.load()`` to read the current lane values back.

The example fills the distributed array in a first pass and reads it back in a second pass before writing the result
to global memory.

  .. literalinclude:: ../../snippets/example/240_lockstep.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-lockstepKernel
    :end-before: END-TUTORIAL-lockstepKernel
    :dedent:

The kernel is launched like any other kernel. Because ``var()`` derives the number of slots from the worker count,
the worker count must be compile-time known; pass a compile-time frame extent such as ``CVec<uint32_t, 8u>{}``.

  .. literalinclude:: ../../snippets/example/240_lockstep.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-lockstepLaunch
    :end-before: END-TUTORIAL-lockstepLaunch
    :dedent:

Choosing the SIMD Width
-----------------------

``scope.concurrent`` has two forms:

- ``scope.concurrent(fn, arg0, ...)`` deduces the SIMD width from the value type of the first data argument.
- ``scope.template concurrent<T_ValueType>(fn, arg0, ...)`` uses the explicitly given value type to select the
  datatype and width of the SIMD mapping. This is useful when the first argument carries no value type (for example
  data wrapped by ``onAcc::map``) or when you want to steer the width deliberately.

  .. literalinclude:: ../../snippets/example/240_lockstep.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-lockstepExplicitWidth
    :end-before: END-TUTORIAL-lockstepExplicitWidth
    :dedent:

Mapped Indexing
---------------

``onAcc::map(data)`` wraps data with an identity mapping and ``onAcc::map(data, offset)`` shifts every per-lane
logical index by ``offset`` before the data is accessed. This makes shifted or neighbor access inside a lockstep
iteration a plain expression instead of manual index arithmetic. The offset must keep accesses in bounds, so reserve
a halo element when reading neighbors.

  .. literalinclude:: ../../snippets/example/240_lockstep.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-lockstepMapped
    :end-before: END-TUTORIAL-lockstepMapped
    :dedent:

API Overview
------------

- ``onAcc::makeLockstep(acc, workGroup, logicalExtent)`` creates the lockstep scope.
- ``scope.var<T>()`` creates the distributed per-worker array.
- ``scope.concurrent(fn, args...)`` / ``scope.template concurrent<T>(fn, args...)`` run the lockstep iteration.
- ``onAcc::map(data)`` and ``onAcc::map(data, offset)`` map logical indices before access.

Practical Advice
----------------

- Use lockstep when you want to keep an intermediate result per element in registers (``var<T>()``) or when you want
  the compiler to pick the SIMD width for you.
- Use ``onAcc::SimdAlgo`` and the ``onHost::concurrent`` / ``onHost::transform`` algorithms when the whole operation
  can be expressed as a host-side algorithm over buffers; they provide the same SIMD portability with less
  boilerplate and are the better default for pure element-wise work.
- Keep the logical extent compile-time known (a ``CVec``) and pass a compile-time frame extent when you use
  ``var()``.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>240_lockstep.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/240_lockstep.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
