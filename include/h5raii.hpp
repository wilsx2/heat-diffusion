#pragma once
#include <H5Ppublic.h>
#include <H5Tpublic.h>
#include <hdf5.h>
#include <span>
#include <string>
#include <string_view>
#include <utility>

template <std::integral I, I Null, typename Deleter>
class UniqueHandle {
protected:
    I _handle;
    Deleter _deleter;

public:
    UniqueHandle(I handle = Null, Deleter deleter = {})
        : _handle{handle}, _deleter{deleter} {}
    UniqueHandle(UniqueHandle &) = delete;
    auto operator=(UniqueHandle &) = delete;
    UniqueHandle(UniqueHandle &&other)
        : _handle{std::exchange(other._handle, Null)} {}
    UniqueHandle &operator=(UniqueHandle &&other) {
        _handle = std::exchange(other._handle, Null);
        return *this;
    }
    ~UniqueHandle() {
        if (*this)
            _deleter(_handle);
    }

    auto release() -> I { return std::exchange(_handle, Null); }
    auto reset() -> void { (void)release(); }
    auto swap(UniqueHandle &other) -> void {
        _handle = std::exchange(other._handle, _handle);
    }

    auto get() const -> I { return _handle; }
    operator bool() const { return _handle != Null; }
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

struct H5ParallelFile : UniqueHandle<hid_t, H5I_INVALID_HID, H5FileCloser> {
    H5ParallelFile(std::string_view filename, MPI_Comm comm) {
        auto plist{H5Pcreate(H5P_FILE_ACCESS)};
        H5Pset_fapl_mpio(plist, comm, MPI_INFO_NULL);
        _handle = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, plist);
        H5Pclose(plist);
    }
};
struct H5Dataspace : UniqueHandle<hid_t, H5I_INVALID_HID, H5DataspaceCloser> {
    H5Dataspace(int ndims, const hsize_t *dims, const hsize_t *maxdims = nullptr)
        : UniqueHandle(H5Screate_simple(ndims, dims, maxdims)) {}
    template <int NDims>
    H5Dataspace(std::span<const hsize_t, NDims> dims,
                std::span<const hsize_t, NDims> maxdims = {})
        : UniqueHandle(H5Screate_simple(NDims, dims.data(), maxdims.data())) {}
};
struct H5Dataset : UniqueHandle<hid_t, H5I_INVALID_HID, H5DatasetCloser> {
    H5Dataset(hid_t file_id, std::string_view name, hid_t type_id,
              hid_t space_id)
        : UniqueHandle(H5Dcreate(file_id, name.data(), type_id, space_id,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) {}
};
