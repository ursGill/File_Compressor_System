"""Core Huffman coding implementation (byte-oriented, independent of the GUI)."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Optional


@dataclass(slots=True)
class HuffmanNode:
    frequency: int
    symbol: Optional[int] = None
    left: Optional["HuffmanNode"] = None
    right: Optional["HuffmanNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.symbol is not None


def frequencies(data: bytes) -> list[int]:
    """Return frequencies for all 256 possible byte values."""
    counts = [0] * 256
    for value in data:
        counts[value] += 1
    return counts


def build_tree(counts: list[int]) -> Optional[HuffmanNode]:
    """Build a deterministic Huffman tree using a min-heap (greedy strategy)."""
    if len(counts) != 256:
        raise ValueError("Frequency table must contain exactly 256 entries.")

    heap: list[tuple[int, int, int, HuffmanNode]] = []
    serial = 0
    for symbol, count in enumerate(counts):
        if count < 0:
            raise ValueError("Frequencies cannot be negative.")
        if count:
            node = HuffmanNode(count, symbol=symbol)
            heapq.heappush(heap, (count, symbol, serial, node))
            serial += 1

    if not heap:
        return None

    while len(heap) > 1:
        freq_a, min_a, _, left = heapq.heappop(heap)
        freq_b, min_b, _, right = heapq.heappop(heap)
        parent = HuffmanNode(freq_a + freq_b, left=left, right=right)
        heapq.heappush(heap, (freq_a + freq_b, min(min_a, min_b), serial, parent))
        serial += 1

    return heap[0][3]


def build_codes(root: Optional[HuffmanNode]) -> dict[int, str]:
    """Map each present byte to its Huffman bit string.

    A single-symbol alphabet uses code '0', avoiding the empty-code edge case.
    """
    if root is None:
        return {}

    result: dict[int, str] = {}

    def visit(node: HuffmanNode, prefix: str) -> None:
        if node.is_leaf:
            result[node.symbol] = prefix or "0"
            return
        if node.left is None or node.right is None:
            raise ValueError("Malformed Huffman tree.")
        visit(node.left, prefix + "0")
        visit(node.right, prefix + "1")

    visit(root, "")
    return result


def encode_data(data: bytes, codes: dict[int, str]) -> tuple[bytes, int]:
    """Encode bytes into packed bits; return (payload, meaningful_bit_count)."""
    bit_buffer = 0
    bit_count_in_buffer = 0
    output = bytearray()
    total_bits = 0

    for symbol in data:
        code = codes.get(symbol)
        if code is None:
            raise ValueError(f"No Huffman code for byte {symbol}.")
        for bit in code:
            bit_buffer = (bit_buffer << 1) | (bit == "1")
            bit_count_in_buffer += 1
            total_bits += 1
            if bit_count_in_buffer == 8:
                output.append(bit_buffer)
                bit_buffer = 0
                bit_count_in_buffer = 0

    if bit_count_in_buffer:
        output.append(bit_buffer << (8 - bit_count_in_buffer))
    return bytes(output), total_bits


def decode_data(
    payload: bytes,
    bit_length: int,
    root: Optional[HuffmanNode],
    original_size: int,
) -> bytes:
    """Decode packed bits and validate the expected output length."""
    if original_size < 0 or bit_length < 0:
        raise ValueError("Negative sizes are invalid.")
    expected_payload_bytes = (bit_length + 7) // 8
    if len(payload) != expected_payload_bytes:
        raise ValueError("Payload length does not match the recorded bit length.")

    if original_size == 0:
        if root is not None or bit_length != 0 or payload:
            raise ValueError("Invalid representation of an empty file.")
        return b""

    if root is None:
        raise ValueError("Missing Huffman tree for non-empty data.")

    # A single-symbol stream is represented by one zero bit per original byte.
    if root.is_leaf:
        if bit_length != original_size:
            raise ValueError("Invalid bit length for a single-symbol file.")
        for index in range(bit_length):
            byte_value = payload[index // 8]
            bit = (byte_value >> (7 - (index % 8))) & 1
            if bit != 0:
                raise ValueError("Invalid bit in single-symbol stream.")
        if bit_length % 8:
            unused = 8 - bit_length % 8
            if payload[-1] & ((1 << unused) - 1):
                raise ValueError("Non-zero padding bits.")
        return bytes([root.symbol]) * original_size

    if bit_length % 8 and payload:
        unused = 8 - bit_length % 8
        if payload[-1] & ((1 << unused) - 1):
            raise ValueError("Non-zero padding bits.")

    result = bytearray()
    node = root
    consumed = 0
    for byte_value in payload:
        for shift in range(7, -1, -1):
            if consumed >= bit_length:
                break
            bit = (byte_value >> shift) & 1
            node = node.right if bit else node.left
            if node is None:
                raise ValueError("Encoded data does not match the Huffman tree.")
            if node.is_leaf:
                result.append(node.symbol)
                if len(result) > original_size:
                    raise ValueError("Decoded output exceeds the recorded original size.")
                node = root
            consumed += 1

    if consumed != bit_length or node is not root:
        raise ValueError("Encoded stream ends in the middle of a Huffman code.")
    if len(result) != original_size:
        raise ValueError(
            f"Decoded {len(result)} bytes; expected {original_size} bytes."
        )
    return bytes(result)


def expected_bit_length(counts: list[int], codes: dict[int, str]) -> int:
    """Calculate the number of meaningful bits implied by frequencies and codes."""
    return sum(count * len(codes[symbol]) for symbol, count in enumerate(counts) if count)
