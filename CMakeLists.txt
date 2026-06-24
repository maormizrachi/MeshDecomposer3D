cmake_minimum_required(VERSION 3.14)
project(MeshDecomposer3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Mostly header-only library with a few .cpp files for explicit template
# instantiations or non-template code.
#
# Consumer must provide include paths for external dependencies:
#   - mpi_utils (cpp-MPI-utils): Serializable, Serializer, exchange, mpi_commands
#   - spatial_ds: OctTree, DistributedOctTree, BoundingBox, Sphere, geometry
