#include "sepal/mapped_file.hpp"

#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#endif

namespace sepal {

namespace {

[[noreturn]] void fail(const std::string& path, const std::string& what) {
    throw std::runtime_error("cannot map '" + path + "': " + what);
}

#if defined(_WIN32)
std::string windows_error(DWORD code) { return "Windows error " + std::to_string(code); }

std::wstring widen(const std::string& utf8, const std::string& path) {
    if (utf8.empty()) return {};
    const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    if (length <= 0) fail(path, "the path is not valid UTF-8");
    std::wstring wide(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), static_cast<int>(utf8.size()), wide.data(), length);
    return wide;
}
#endif

} // namespace

MappedFile::~MappedFile() { release(); }

MappedFile::MappedFile(MappedFile&& other) noexcept { *this = std::move(other); }

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        release();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        path_ = std::move(other.path_);
        file_handle_ = std::exchange(other.file_handle_, nullptr);
        mapping_handle_ = std::exchange(other.mapping_handle_, nullptr);
    }
    return *this;
}

#if defined(_WIN32)

MappedFile MappedFile::open(const std::string& path) {
    MappedFile file;
    file.path_ = path;
    HANDLE handle = CreateFileW(widen(path, path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) fail(path, "cannot open the file (" + windows_error(GetLastError()) + ")");
    file.file_handle_ = handle;
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(handle, &size)) fail(path, "cannot read the file size (" + windows_error(GetLastError()) + ")");
    if (static_cast<unsigned long long>(size.QuadPart) > static_cast<unsigned long long>(static_cast<std::size_t>(-1))) {
        fail(path, "the file is larger than the address space");
    }
    file.size_ = static_cast<std::size_t>(size.QuadPart);
    if (file.size_ == 0) return file; // a zero-length file cannot be mapped; it is an empty span
    HANDLE mapping = CreateFileMappingW(handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) fail(path, "cannot create the mapping (" + windows_error(GetLastError()) + ")");
    file.mapping_handle_ = mapping;
    void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) fail(path, "cannot map the view (" + windows_error(GetLastError()) + ")");
    file.data_ = static_cast<const unsigned char*>(view);
    return file;
}

void MappedFile::release() noexcept {
    if (data_ != nullptr) UnmapViewOfFile(data_);
    if (mapping_handle_ != nullptr) CloseHandle(static_cast<HANDLE>(mapping_handle_));
    if (file_handle_ != nullptr) CloseHandle(static_cast<HANDLE>(file_handle_));
    data_ = nullptr;
    size_ = 0;
    mapping_handle_ = nullptr;
    file_handle_ = nullptr;
}

#else

MappedFile MappedFile::open(const std::string& path) {
    MappedFile file;
    file.path_ = path;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fail(path, std::string("cannot open the file (") + std::strerror(errno) + ")");
    struct stat info{};
    if (::fstat(fd, &info) != 0) {
        const std::string reason = std::strerror(errno);
        ::close(fd);
        fail(path, "cannot read the file size (" + reason + ")");
    }
    if (!S_ISREG(info.st_mode)) {
        ::close(fd);
        fail(path, "not a regular file");
    }
    file.size_ = static_cast<std::size_t>(info.st_size);
    if (file.size_ > 0) {
        void* view = ::mmap(nullptr, file.size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (view == MAP_FAILED) {
            const std::string reason = std::strerror(errno);
            ::close(fd);
            fail(path, "mmap failed (" + reason + ")");
        }
        file.data_ = static_cast<const unsigned char*>(view);
    }
    ::close(fd); // the mapping stays valid without the descriptor
    return file;
}

void MappedFile::release() noexcept {
    if (data_ != nullptr) ::munmap(const_cast<unsigned char*>(data_), size_);
    data_ = nullptr;
    size_ = 0;
}

#endif

} // namespace sepal
