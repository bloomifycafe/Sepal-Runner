// A file mapped into memory read-only. Windows (CreateFileMapping) and POSIX (mmap) behind one small class; no
// third-party dependency. The bytes are never copied, so a checkpoint of many gigabytes costs address space and
// not memory until a page is read.
#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace sepal {

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    /// Maps the whole file. The path is UTF-8. Throws std::runtime_error naming the path when the file cannot
    /// be opened or mapped. An empty file maps to an empty span.
    static MappedFile open(const std::string& path);

    /// The file's bytes, valid while this object lives. The first byte is aligned to at least a page.
    std::span<const unsigned char> bytes() const { return {data_, size_}; }
    const std::string& path() const { return path_; }

private:
    void release() noexcept;

    const unsigned char* data_ = nullptr;
    std::size_t size_ = 0;
    std::string path_;
    void* file_handle_ = nullptr;    // Windows: the file handle; unused elsewhere
    void* mapping_handle_ = nullptr; // Windows: the mapping object; unused elsewhere
};

} // namespace sepal
