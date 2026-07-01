#ifndef MESH_DECOMPOSER_EXCHANGE_FIELD_IO_HPP
#define MESH_DECOMPOSER_EXCHANGE_FIELD_IO_HPP

#ifdef RICH_MPI

#include <type_traits>

#include <mpi_utils/serialize/Serializable.hpp>
#include <mpi_utils/serialize/Serializer.hpp>

namespace points_manager_detail
{

template<typename T>
inline constexpr bool is_serializable_v = std::is_base_of_v<Serializable, T>;

template<typename T>
inline constexpr bool is_exchange_field_v =
    is_serializable_v<T> || std::is_trivially_copyable_v<T>;

template<typename T, typename = void>
struct ExchangeFieldIO;

template<typename T>
struct ExchangeFieldIO<T, std::enable_if_t<is_serializable_v<T>>>
{
    static size_t dump(const T &field, Serializer *serializer)
    {
        return field.dump(serializer);
    }

    static size_t load(T &field, const Serializer *serializer, size_t byteOffset)
    {
        return field.load(serializer, byteOffset);
    }
};

template<typename T>
struct ExchangeFieldIO<T, std::enable_if_t<!is_serializable_v<T> && std::is_trivially_copyable_v<T>>>
{
    static size_t dump(const T &field, Serializer *serializer)
    {
        return serializer->insert(field);
    }

    static size_t load(T &field, const Serializer *serializer, size_t byteOffset)
    {
        return serializer->extract(field, byteOffset);
    }
};

} // namespace points_manager_detail

#endif // RICH_MPI

#endif // MESH_DECOMPOSER_EXCHANGE_FIELD_IO_HPP
