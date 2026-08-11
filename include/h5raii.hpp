#pragma once
#include <H5Ppublic.h>
#include <H5Tpublic.h>
#include <hdf5.h>
#include <span>
#include <string>
#include <string_view>
#include <utility>

template <std::integral I, typename Deleter>
class UniqueHandle {
private:
    I _handle;
    Deleter _deleter;

public:
    UniqueHandle(I handle = NULL, Deleter deleter = {})
        : _handle{handle}, _deleter{deleter} {}
    UniqueHandle(UniqueHandle &) = delete;
    auto operator=(UniqueHandle &) = delete;
    UniqueHandle(UniqueHandle &&other)
        : _handle{std::exchange(other._handle, NULL)} {}
    UniqueHandle &operator=(UniqueHandle &&other) {
        _handle = std::exchange(other._handle, NULL);
        return *this;
    }
    ~UniqueHandle() { _deleter(_handle); }

    auto release() -> I { return std::exchange(_handle, NULL); }
    auto reset() -> void { (void)release(); }
    auto swap(UniqueHandle &other) -> void {
        _handle = std::exchange(other._handle, _handle);
    }

    auto get() const -> I { return _handle; }
    operator bool() const { return _handle == NULL; }
    auto operator*() const -> I { return _handle; }
};

struct H5FileCloser {
    void operator()(hid_t id) { H5Fclose(id); }
};
struct H5DataspaceCloser {
    void operator()(hid_t id) { H5Sclose(id); }
};
struct H5DatasetCloser {
    void operator()(hid_t id) { H5Dclose(id); }
};
struct H5File : UniqueHandle<hid_t, H5FileCloser> {
    H5File(std::string_view filename)
        : UniqueHandle(H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT,
                                 H5P_DEFAULT)) {}
};
struct H5Dataspace : UniqueHandle<hid_t, H5DataspaceCloser> {
    H5Dataspace(int ndims, const hsize_t *dims, const hsize_t *maxdims = NULL)
        : UniqueHandle(H5Screate_simple(ndims, dims, maxdims)) {}
    template<int NDims>
    H5Dataspace(std::span<const hsize_t, NDims> dims, std::span<const hsize_t, NDims> maxdims = {})
        : UniqueHandle(H5Screate_simple(NDims, dims.data(), maxdims.data())) {}
};
struct H5Dataset : UniqueHandle<hid_t, H5DatasetCloser> {
    H5Dataset(hid_t file_id, hid_t ds_id, std::string_view name)
        : UniqueHandle(H5Dcreate(file_id, name.data(), H5T_STD_I32BE, ds_id,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) {}
};
