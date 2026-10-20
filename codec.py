"""HUF1 file container and file-level compression/decompression operations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import zlib

from huffman import (
    build_codes, build_tree, decode_data, encode_data, expected_bit_length, frequencies
)

MAGIC = b"HUF1"
VERSION = 1
# magic, version, original size, CRC32 of original data, meaningful bit length, symbol count
HEADER = struct.Struct(">4sBQIQH")
FREQUENCY_ENTRY = struct.Struct(">BQ")
MAX_SYMBOLS = 256


@dataclass
class CompressionStats:
    source_path: str
    output_path: str
    original_size: int
    compressed_size: int
    compression_ratio_percent: float
    space_saved_percent: float
    saved_bytes: int
    symbol_count: int
    bit_length: int

    @classmethod
    def create(cls, source: Path, output: Path, original_size: int,
               compressed_size: int, symbol_count: int, bit_length: int):
        ratio = (compressed_size / original_size * 100) if original_size else 0.0
        saved = original_size - compressed_size
        saved_percent = (saved / original_size * 100) if original_size else 0.0
        return cls(str(source), str(output), original_size, compressed_size,
                   ratio, saved_percent, saved, symbol_count, bit_length)


def _write_atomically(output_path: Path, data: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(output_path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def compress_file(source_path: str | Path, output_path: str | Path) -> tuple[CompressionStats, dict, object]:
    source = Path(source_path)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source file does not exist: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must be different.")

    data = source.read_bytes()
    counts = frequencies(data)
    root = build_tree(counts)
    codes = build_codes(root)
    payload, bit_length = encode_data(data, codes)
    if bit_length != expected_bit_length(counts, codes):
        raise RuntimeError("Internal error: encoded bit length mismatch.")

    symbols = [(symbol, count) for symbol, count in enumerate(counts) if count]
    header = HEADER.pack(MAGIC, VERSION, len(data), zlib.crc32(data) & 0xFFFFFFFF,
                         bit_length, len(symbols))
    table = b"".join(FREQUENCY_ENTRY.pack(symbol, count) for symbol, count in symbols)
    container = header + table + payload
    _write_atomically(output, container)

    stats = CompressionStats.create(source, output, len(data), len(container),
                                    len(symbols), bit_length)
    return stats, codes, root


def decompress_file(source_path: str | Path, output_path: str | Path) -> CompressionStats:
    source = Path(source_path)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Compressed file does not exist: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must be different.")

    blob = source.read_bytes()
    if len(blob) < HEADER.size:
        raise ValueError("File is too short to contain a HUF header.")

    magic, version, original_size, expected_crc, bit_length, symbol_count = HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise ValueError("Not a supported HUF file (missing HUF1 signature).")
    if version != VERSION:
        raise ValueError(f"Unsupported HUF format version: {version}")
    if symbol_count > MAX_SYMBOLS:
        raise ValueError("Invalid symbol count in header.")

    offset = HEADER.size
    table_end = offset + symbol_count * FREQUENCY_ENTRY.size
    if table_end > len(blob):
        raise ValueError("Truncated frequency table.")

    counts = [0] * 256
    seen = set()
    for _ in range(symbol_count):
        symbol, count = FREQUENCY_ENTRY.unpack_from(blob, offset)
        offset += FREQUENCY_ENTRY.size
        if symbol in seen:
            raise ValueError("Duplicate symbol in frequency table.")
        if count == 0:
            raise ValueError("Frequency table contains a zero frequency.")
        seen.add(symbol)
        counts[symbol] = count

    if sum(counts) != original_size:
        raise ValueError("Frequency table does not match the recorded original size.")
    if original_size == 0 and symbol_count != 0:
        raise ValueError("Empty files must not contain symbols.")
    if original_size > 0 and symbol_count == 0:
        raise ValueError("Non-empty files must contain at least one symbol.")

    payload = blob[table_end:]
    if len(payload) != (bit_length + 7) // 8:
        raise ValueError("Compressed payload size does not match the header.")

    root = build_tree(counts)
    codes = build_codes(root)
    if expected_bit_length(counts, codes) != bit_length:
        raise ValueError("Recorded bit length does not match the frequency table.")

    decoded = decode_data(payload, bit_length, root, original_size)
    actual_crc = zlib.crc32(decoded) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError("Checksum mismatch: file may be corrupted.")

    _write_atomically(output, decoded)
    return CompressionStats.create(source, output, len(decoded), len(blob),
                                    symbol_count, bit_length)


def verify_files(original_path: str | Path, decoded_path: str | Path) -> bool:
    """Compare byte-for-byte; checksum validation also happens during decompression."""
    return Path(original_path).read_bytes() == Path(decoded_path).read_bytes()


def default_compressed_path(source_path: str | Path) -> Path:
    source = Path(source_path)
    return source.with_name(source.name + ".huf")


def default_decompressed_path(source_path: str | Path) -> Path:
    source = Path(source_path)
    if source.suffix.lower() == ".huf":
        return source.with_suffix("")
    return source.with_name(source.name + ".decompressed")
