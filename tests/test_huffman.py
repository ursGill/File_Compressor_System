import tempfile
import unittest
from pathlib import Path

from huffman import frequencies, build_tree, build_codes, encode_data, decode_data, expected_bit_length
from codec import compress_file, decompress_file, verify_files, HEADER, MAGIC


class HuffmanAlgorithmTests(unittest.TestCase):
    def round_trip(self, data: bytes):
        counts = frequencies(data)
        root = build_tree(counts)
        codes = build_codes(root)
        payload, bit_length = encode_data(data, codes)
        self.assertEqual(bit_length, expected_bit_length(counts, codes))
        self.assertEqual(decode_data(payload, bit_length, root, len(data)), data)

    def test_empty_input(self):
        self.round_trip(b"")

    def test_single_symbol(self):
        self.round_trip(b"AAAAAA")

    def test_text(self):
        self.round_trip(b"this is an example for huffman encoding")

    def test_all_byte_values(self):
        self.round_trip(bytes(range(256)))

    def test_binary_data(self):
        self.round_trip(bytes([0, 255, 0, 128, 10, 13, 255, 128]) * 20)


class FileFormatTests(unittest.TestCase):
    def test_file_round_trips(self):
        samples = [
            b"",
            b"x",
            b"Z" * 200,
            b"hello Huffman!",
            bytes(range(256)) * 3,
            bytes([0, 255, 128, 10, 0, 255]) * 100,
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index, data in enumerate(samples):
                original = root / f"sample_{index}.bin"
                packed = root / f"sample_{index}.huf"
                restored = root / f"restored_{index}.bin"
                original.write_bytes(data)
                stats, _, _ = compress_file(original, packed)
                decompress_file(packed, restored)
                self.assertEqual(restored.read_bytes(), data)
                self.assertTrue(verify_files(original, restored))
                self.assertEqual(stats.original_size, len(data))

    def test_bad_magic_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bad = root / "bad.huf"
            out = root / "out.bin"
            bad.write_bytes(b"NOPE" + b"\x00" * 30)
            with self.assertRaisesRegex(ValueError, "signature"):
                decompress_file(bad, out)

    def test_truncated_header_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bad = root / "short.huf"
            bad.write_bytes(b"HUF1")
            with self.assertRaisesRegex(ValueError, "too short"):
                decompress_file(bad, root / "out.bin")

    def test_corrupted_payload_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "original.txt"
            packed = root / "original.huf"
            restored = root / "restored.txt"
            original.write_bytes(b"some repeated text some repeated text")
            compress_file(original, packed)
            data = bytearray(packed.read_bytes())
            data[-1] ^= 0x80
            packed.write_bytes(data)
            with self.assertRaises(ValueError):
                decompress_file(packed, restored)

    def test_input_output_same_path_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "file.txt"
            source.write_text("safe", encoding="utf-8")
            with self.assertRaises(ValueError):
                compress_file(source, source)


if __name__ == "__main__":
    unittest.main()
