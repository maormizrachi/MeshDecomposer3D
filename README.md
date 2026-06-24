# MeshDecomposer3D

A C++17 header-only library for 3D domain decomposition and load balancing.

## Overview

MeshDecomposer3D distributes 3D point sets across MPI ranks using space-filling curves (Hilbert). It provides load balancing, MPI point exchange, and spatial queries for distributed simulations. The library is templated on a user-defined point type and an optional per-point payload, allowing it to integrate with a wide range of particle or mesh-based codes.

Typical use cases include assigning ownership of Lagrangian particles, rebalancing workloads as point distributions evolve, and querying which MPI ranks intersect a spatial region.

## Features

- Hilbert curve-based spatial partitioning and load balancing
- Templated on point type (`PointT`) and exchange payload (`PayloadT`)
- 16 coordinate transformation kernels for complex geometries
- Environment agents for spatial rank intersection queries
- Distributed point management with automatic rebalancing
- Weighted balance algorithms

## Architecture

| Directory | Description |
|-----------|-------------|
| `load_balancing/` | `LoadBalancer` hierarchy: base class, `CurveLoadBalancer`, and `HilbertLoadBalancer` for curve-indexed domain assignment |
| `points_manager/` | `PointsManager` and `HilbertPointsManager` for distributed point exchange and rebalancing |
| `environment/` | `EnvironmentAgent` hierarchy for owner lookup and sphere–rank intersection queries |
| `hilbert/` | Hilbert curve ordering, convertors, and rectangular/ordinary variants |
| `kernels/` | `IndexingKernel3D` interface and 16 concrete coordinate transformation kernels |
| `balance/` | Weighted partitioning algorithms (`weightedBalance`, `weightedBalance2`, `weightedBalance3`) |
| `kernels/math/` | `Mat33` and `Mat44` matrix utilities used by transformation kernels |

### Directory contents

**`hilbert/`**
- `HilbertOrder3D.hpp`, `HilbertOrder3D_Utils.hpp`, `HilbertConvertor3D.hpp`, `hilbertTypes.h`
- `ordinary/HilbertOrdinaryConvertor3D.hpp`
- `rectangular/HilbertRectangularConvertor3D.hpp`, `HilbertRectangularTree3D.hpp`

**`kernels/`**
- `IndexingKernel3D.hpp` (base interface)
- `Identity`, `Move`, `Scale`, `GenericScale`, `Shrink`, `Linear`, `Affine`, `Rotation`, `Reflection`, `Sphere`, `Rectangle`, `SameRectangle`, `RectangleShrink`, `Frustrum`, `RevedFrustrum`, `Parallelepiped`
- `kernel_detail.hpp`, `math/Mat33.hpp`, `math/Mat44.hpp`

**`balance/`**
- `balance.hpp`, `weightedBalance.hpp`, `weightedBalance2.hpp`, `weightedBalance3.hpp`

**`environment/`**
- `EnvironmentAgent.hpp`, `CurveEnvAgent.hpp`, `PlainDistributedOctEnvAgent.hpp`
- `hilbert/HilbertCurveEnvAgent.hpp`, `HilbertTreeEnvAgent.hpp`, `DistributedOctEnvAgent.hpp`

## Requirements

- **C++17** compiler
- **MPI**
- **External dependencies** (include paths must be provided by the consumer):
  - [cpp-MPI-utils](https://github.com/maormizrachi/cpp-MPI-utils) (`mpi_utils`): serialization, MPI exchange utilities
  - `spatial_ds`: octrees, bounding boxes, distributed spatial data structures

Define `RICH_MPI` when compiling code that uses MPI-dependent components (`PointsManager`, `HilbertPointsManager`, environment agents).

## Point Type Requirements

`PointT` must satisfy the interface documented in `point_utils.hpp`:

| Requirement | Description |
|-------------|-------------|
| `coord_type` | Scalar type alias (e.g. `double`) |
| `x`, `y`, `z` | Public coordinate members |
| `operator[](size_t)` | Const and non-const element access |
| Constructors | Default and three-argument `(x, y, z)` |
| Operators | `-`, `*`, `==`, `!=` as documented |

Free functions (via ADL or in the global namespace):

```cpp
abs(PointT)                    // Euclidean norm
ScalarProd(PointT, PointT)     // dot product
CrossProduct(PointT, PointT)   // cross product
normalize(PointT)              // unit vector
```

## Payload Type Requirements

`PayloadT` must be either:

- Derived from `Serializable` (from `mpi_utils`), supporting `dump`/`load`, or
- Trivially copyable (serialized via raw byte insert/extract)

Both `PointT` and `PayloadT` are checked at compile time when used in `ExchangePoint`.

## Usage

```cpp
#define RICH_MPI
#include <mpi.h>
#include "points_manager/HilbertPointsManager.hpp"

struct Vec3 {
    using coord_type = double;
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    coord_type& operator[](size_t i) { return (&x)[i]; }
    const coord_type& operator[](size_t i) const { return (&x)[i]; }
};
// Provide abs, ScalarProd, CrossProduct, normalize for Vec3.

struct ParticleData { int id = 0; /* or inherit Serializable */ };

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    Vec3 ll{0, 0, 0}, ur{1, 1, 1};
    HilbertPointsManager<Vec3, ParticleData> manager(ll, ur);

    std::vector<Vec3> points = /* local or global input */;
    std::vector<double> weights(points.size(), 1.0);
    std::vector<ParticleData> payloads(points.size());
    std::vector<size_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);

    // First call: all points on each rank; sizes of points and indices must match.
    auto result = manager.exchange(points, weights, payloads, indices, /*noExchange=*/false);

    // result.newPoints, result.newPayloads, result.newIndices hold the local set after exchange.

    // Subsequent timesteps: use update() to rebalance and exchange as needed.
    // auto result2 = manager.update(points, weights, payloads, indices);

    MPI_Finalize();
    return 0;
}
```

Optional geometry mapping before Hilbert indexing:

```cpp
#include "kernels/Sphere.hpp"
manager.setIndexing(std::make_shared<const Kernelization3D::Sphere<Vec3>>(center, radius));
```

Query spatial neighbors after initialization:

```cpp
auto env = manager.getEnvironmentAgent();
auto ranks = env->getIntersectingRanks(center, radius);
int owner = env->getOwner(point);
```

## Error Handling

Errors are reported through `DomainDecompError`, a subclass of `std::runtime_error`. Contextual key–value entries can be attached before throwing:

```cpp
DomainDecompError err("HilbertPointsManager::exchange: size mismatch");
err.addEntry("allPoints.size()", allPoints.size());
err.addEntry("indicesToWorkWith.size()", indicesToWorkWith.size());
throw err;
```

The `what()` message includes the base message followed by all attached entries.

## Building

The library is primarily header-only. Add the repository root to your include path and link against MPI. Ensure include paths for `mpi_utils` and `spatial_ds` are available to the compiler.

```cmake
target_compile_definitions(my_target PRIVATE RICH_MPI)
target_include_directories(my_target PRIVATE
    /path/to/MeshDecomposer3D
    /path/to/cpp-MPI-utils
    /path/to/spatial_ds
)
find_package(MPI REQUIRED)
target_link_libraries(my_target PRIVATE MPI::MPI_CXX)
```

## License

[License placeholder — specify license here.]
