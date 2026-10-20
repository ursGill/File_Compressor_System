"""CustomTkinter dark GUI for the Huffman File Compressor."""
from __future__ import annotations

import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk

from codec import (
    compress_file, decompress_file, verify_files,
    default_compressed_path, default_decompressed_path
)
from huffman import frequencies, build_tree, build_codes, HuffmanNode


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def byte_label(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{value} B"
        amount /= 1024
    return f"{value} B"


def make_tree_lines(root: HuffmanNode | None) -> str:
    if root is None:
        return "(empty file: no Huffman tree)"

    lines: list[str] = []
    def walk(node: HuffmanNode, prefix: str, edge: str) -> None:
        if node.is_leaf:
            label = f"byte={node.symbol} ({repr(bytes([node.symbol]))[2:-1]})"
        else:
            label = "internal"
        lines.append(f"{prefix}{edge}{label}  [freq={node.frequency}]")
        if node.left:
            walk(node.left, prefix + ("    " if edge else ""), "0 -> ")
        if node.right:
            walk(node.right, prefix + ("    " if edge else ""), "1 -> ")
    walk(root, "", "")
    return "\n".join(lines)


class HuffmanCompressorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Huffman File Compressor | DAA Project")
        self.geometry("1080x760")
        self.minsize(900, 640)
        self.selected_files: list[Path] = []
        self.last_analysis = None
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self, corner_radius=16)
        header.grid(row=0, column=0, padx=18, pady=(18, 10), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="HUFFMAN FILE COMPRESSOR",
                     font=ctk.CTkFont(size=24, weight="bold")).grid(
                         row=0, column=0, padx=18, pady=(14, 2), sticky="w")
        ctk.CTkLabel(header, text="DAA Project • Greedy Algorithm • Lossless Compression",
                     text_color="#A7B0C0").grid(row=1, column=0, padx=18, pady=(0, 14), sticky="w")

        controls = ctk.CTkFrame(self, corner_radius=14)
        controls.grid(row=1, column=0, padx=18, pady=8, sticky="ew")
        controls.grid_columnconfigure(0, weight=1)
        self.file_summary = ctk.CTkLabel(controls, text="No files selected", anchor="w")
        self.file_summary.grid(row=0, column=0, padx=14, pady=12, sticky="ew")
        ctk.CTkButton(controls, text="Add files", command=self.add_files, width=110).grid(
            row=0, column=1, padx=5, pady=10)
        ctk.CTkButton(controls, text="Clear", command=self.clear_files, width=75,
                      fg_color="#454B57", hover_color="#343943").grid(
            row=0, column=2, padx=(5, 12), pady=10)

        main = ctk.CTkTabview(self, corner_radius=14)
        main.grid(row=2, column=0, padx=18, pady=8, sticky="nsew")
        self.operation_tab = main.add("Compress / Decompress")
        self.analysis_tab = main.add("Huffman Analysis")
        self.stats_tab = main.add("Statistics")
        self._build_operations()
        self._build_analysis()
        self._build_stats()

        self.status = ctk.CTkLabel(self, text="Ready", anchor="w", text_color="#A7B0C0")
        self.status.grid(row=3, column=0, padx=22, pady=(3, 14), sticky="ew")

    def _build_operations(self):
        tab = self.operation_tab
        tab.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(tab, text="Choose an operation", font=ctk.CTkFont(size=19, weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=14, pady=(18, 12), sticky="w")
        ctk.CTkLabel(tab, text="Compression creates a .huf file. Decompression restores the original bytes.",
                     text_color="#A7B0C0", wraplength=800).grid(
            row=1, column=0, columnspan=2, padx=14, pady=(0, 16), sticky="w")

        ctk.CTkButton(tab, text="Compress selected file(s)", height=42,
                      command=lambda: self._run_batch("compress")).grid(
            row=2, column=0, padx=14, pady=8, sticky="ew")
        ctk.CTkButton(tab, text="Decompress selected .huf file(s)", height=42,
                      command=lambda: self._run_batch("decompress"),
                      fg_color="#267A68", hover_color="#1E6254").grid(
            row=2, column=1, padx=14, pady=8, sticky="ew")
        ctk.CTkButton(tab, text="Batch compress (all selected)", height=38,
                      command=lambda: self._run_batch("compress")).grid(
            row=3, column=0, padx=14, pady=8, sticky="ew")
        ctk.CTkButton(tab, text="Batch decompress (all selected)", height=38,
                      command=lambda: self._run_batch("decompress"),
                      fg_color="#267A68", hover_color="#1E6254").grid(
            row=3, column=1, padx=14, pady=8, sticky="ew")

        self.progress = ctk.CTkProgressBar(tab)
        self.progress.grid(row=4, column=0, columnspan=2, padx=14, pady=(25, 8), sticky="ew")
        self.progress.set(0)
        ctk.CTkLabel(tab, text="Verification compares original and decompressed files byte-for-byte.",
                     text_color="#A7B0C0").grid(
            row=5, column=0, columnspan=2, padx=14, pady=6, sticky="w")
        ctk.CTkButton(tab, text="Verify original vs decompressed…",
                      command=self._verify_dialog, fg_color="#454B57",
                      hover_color="#343943").grid(
            row=6, column=0, columnspan=2, padx=14, pady=10, sticky="ew")

    def _build_analysis(self):
        tab = self.analysis_tab
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(tab, text="Huffman Tree and Codes",
                     font=ctk.CTkFont(size=19, weight="bold")).grid(
            row=0, column=0, padx=12, pady=(12, 8), sticky="w")
        self.analysis_text = ctk.CTkTextbox(tab, font=("Consolas", 12), wrap="none")
        self.analysis_text.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.analysis_text.insert("1.0", "Select a file, then click “Build analysis”.")
        self.analysis_text.configure(state="disabled")
        ctk.CTkButton(tab, text="Build analysis for selected file",
                      command=self._build_analysis_for_file).grid(
            row=2, column=0, padx=12, pady=(0, 12), sticky="ew")

    def _build_stats(self):
        tab = self.stats_tab
        tab.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(tab, text="Latest operation statistics",
                     font=ctk.CTkFont(size=19, weight="bold")).grid(
            row=0, column=0, padx=12, pady=(12, 8), sticky="w")
        self.stats_text = ctk.CTkTextbox(tab, font=("Consolas", 13))
        self.stats_text.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        tab.grid_rowconfigure(1, weight=1)
        self.stats_text.insert("1.0", "No operation completed yet.")
        self.stats_text.configure(state="disabled")

    def _set_text(self, widget, value: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def add_files(self):
        names = filedialog.askopenfilenames(title="Select files to process")
        if not names:
            return
        for name in names:
            path = Path(name)
            if path not in self.selected_files:
                self.selected_files.append(path)
        self._update_file_summary()

    def clear_files(self):
        self.selected_files.clear()
        self._update_file_summary()
        self.status.configure(text="Selection cleared.")

    def _update_file_summary(self):
        if not self.selected_files:
            text = "No files selected"
        else:
            names = [p.name for p in self.selected_files]
            text = f"{len(names)} file(s): " + ", ".join(names[:3])
            if len(names) > 3:
                text += f" and {len(names) - 3} more"
        self.file_summary.configure(text=text)

    def _run_batch(self, operation: str):
        if not self.selected_files:
            messagebox.showinfo("Select files", "Add one or more files first.")
            return

        # Decompression only accepts this application's .huf container files.
        # If the user selected both the original and compressed file, ignore
        # non-.huf files instead of trying to decompress them and showing an error.
        if operation == "decompress":
            files = [p for p in self.selected_files if p.suffix.lower() == ".huf"]
            skipped = [p for p in self.selected_files if p.suffix.lower() != ".huf"]
            if not files:
                messagebox.showwarning(
                    "No compressed files selected",
                    "Select a file created by this application (ending in .huf).\\n\\n"
                    "Original files such as .mp4, .txt, or .jpg cannot be decompressed directly."
                )
                return
            if skipped:
                self.status.configure(
                    text=f"Decompressing {len(files)} .huf file(s); skipped {len(skipped)} non-.huf file(s)."
                )
        else:
            files = list(self.selected_files)

        self.progress.set(0)
        self.status.configure(text=f"Starting {operation}…")
        threading.Thread(target=self._batch_worker, args=(files, operation), daemon=True).start()

    def _batch_worker(self, files: list[Path], operation: str):
        outcomes = []
        total = len(files)
        for index, source in enumerate(files, start=1):
            try:
                if operation == "compress":
                    output = default_compressed_path(source)
                    stats, codes, root = compress_file(source, output)
                    outcomes.append((True, source, output, stats, codes, root, None))
                else:
                    output = default_decompressed_path(source)
                    stats = decompress_file(source, output)
                    outcomes.append((True, source, output, stats, None, None, None))
            except Exception as exc:
                outcomes.append((False, source, None, None, None, None, str(exc)))
            self.after(0, self.progress.set, index / total)
        self.after(0, self._finish_batch, operation, outcomes)

    def _finish_batch(self, operation, outcomes):
        lines = [f"Operation: {operation.upper()}", f"Files attempted: {len(outcomes)}", ""]
        errors = []
        for success, source, output, stats, codes, root, error in outcomes:
            if success:
                lines.append(f"SUCCESS  {source.name} -> {output.name}")
                lines.append(f"  Input: {byte_label(stats.original_size)} | Output: {byte_label(stats.compressed_size)}")
                if operation == "compress":
                    lines.append(f"  Compressed/original: {stats.compression_ratio_percent:.2f}%")
                    lines.append(f"  Space saved: {stats.space_saved_percent:.2f}% ({stats.saved_bytes} bytes)")
                    self.last_analysis = (source, codes, root)
                else:
                    lines.append("  Integrity: CRC32 verified")
                lines.append("")
            else:
                lines.append(f"FAILED   {source.name}: {error}")
                errors.append(f"{source.name}: {error}")
        self._set_text(self.stats_text, "\n".join(lines))
        self.status.configure(text=f"Finished: {len(outcomes)-len(errors)} succeeded, {len(errors)} failed.")
        if errors:
            messagebox.showwarning("Some files failed", "\n".join(errors[:8]))

    def _build_analysis_for_file(self):
        if not self.selected_files:
            messagebox.showinfo("Select a file", "Add a file first.")
            return
        source = self.selected_files[0]
        try:
            data = source.read_bytes()
            counts = frequencies(data)
            root = build_tree(counts)
            codes = build_codes(root)
            lines = [
                f"File: {source}",
                f"Size: {len(data)} bytes",
                f"Distinct byte values: {sum(1 for n in counts if n)}",
                "",
                "HUFFMAN TREE (0 = left edge, 1 = right edge)",
                "-" * 54,
                make_tree_lines(root),
                "",
                "BYTE CODES",
                "-" * 54,
                "BYTE   HEX   FREQUENCY   CODE",
            ]
            for symbol in sorted(codes):
                printable = repr(bytes([symbol]))
                lines.append(f"{symbol:>4}   0x{symbol:02X}   {counts[symbol]:>9}   {codes[symbol]}   {printable}")
            if not codes:
                lines.append("(empty file: no codes)")
            self._set_text(self.analysis_text, "\n".join(lines))
            self.last_analysis = (source, codes, root)
            self.status.configure(text=f"Analysis built for {source.name}.")
        except Exception as exc:
            messagebox.showerror("Analysis error", str(exc))

    def _verify_dialog(self):
        original = filedialog.askopenfilename(title="Choose original file")
        if not original:
            return
        decoded = filedialog.askopenfilename(title="Choose decompressed file")
        if not decoded:
            return
        try:
            matched = verify_files(original, decoded)
            if matched:
                messagebox.showinfo("Verification passed", "The files match byte-for-byte.")
            else:
                messagebox.showwarning("Verification failed", "The files differ.")
        except Exception as exc:
            messagebox.showerror("Verification error", str(exc))


if __name__ == "__main__":
    HuffmanCompressorApp().mainloop()
