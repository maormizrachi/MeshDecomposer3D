#ifndef MESH_DECOMPOSER_RUNTIME_LOG_HPP
#define MESH_DECOMPOSER_RUNTIME_LOG_HPP

#include <cstdlib>
#include <cstring>

namespace mesh_decomposer_runtime_log_detail
{
inline bool Detailed()
{
    char const* const value = std::getenv("RICH_RUNTIME_LOG");
    return value != nullptr && std::strcmp(value, "detailed") == 0;
}
}

#endif // MESH_DECOMPOSER_RUNTIME_LOG_HPP
