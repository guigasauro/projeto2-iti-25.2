import os
import re
import subprocess
import tempfile
from dataclasses import dataclass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "modelos_ppm")
DEFAULT_PPM_CMD = os.path.abspath(os.path.join(BASE_DIR, "..", "ppm"))
DEFAULT_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class PPMConfig:
    kmax: int = 5
    window: int = 1000
    threshold: float = 10.0
    reset: int = 0

    def args(self):
        return (
            str(self.kmax),
            str(self.window),
            f"{self.threshold:g}",
            str(self.reset),
        )

    def to_dict(self):
        return {
            "kmax": self.kmax,
            "window": self.window,
            "threshold": self.threshold,
            "reset": self.reset,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            kmax=int(data["kmax"]),
            window=int(data["window"]),
            threshold=float(data["threshold"]),
            reset=int(data["reset"]),
        )


def sanitize_label(label):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return safe or "classe"


def remove_artifacts(*paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def collect_class_files(
    dataset_dir,
    recursive=False,
    include_completos=False,
    limit_per_class=0,
):
    grouped = {}

    for class_name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        files = []

        if recursive:
            for root, dirs, names in os.walk(class_dir):
                if not include_completos:
                    dirs[:] = [name for name in dirs if name.lower() != "completos"]

                for name in sorted(names):
                    path = os.path.join(root, name)
                    if name.lower().endswith(".txt") and os.path.isfile(path):
                        files.append(os.path.abspath(path))
        else:
            for name in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, name)
                if name.lower().endswith(".txt") and os.path.isfile(path):
                    files.append(os.path.abspath(path))

        if limit_per_class > 0:
            files = files[:limit_per_class]

        if files:
            grouped[class_name] = files

    return grouped


def read_text_bytes(path, lowercase=False):
    if lowercase:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read().lower()
        return text.encode("utf-8")

    with open(path, "rb") as file:
        return file.read()


def read_text_for_chunking(path, lowercase=False):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()
    return text.lower() if lowercase else text


def build_model_bytes(paths, separator, lowercase=False, max_bytes=0):
    separator_bytes = separator.encode("utf-8")
    parts = []
    selected_paths = []
    total_bytes = 0

    for path in paths:
        data = read_text_bytes(path, lowercase=lowercase)
        addition = data if not parts else separator_bytes + data

        if max_bytes > 0 and parts and total_bytes + len(addition) > max_bytes:
            break

        parts.append(addition)
        selected_paths.append(path)
        total_bytes += len(addition)

        if max_bytes > 0 and total_bytes >= max_bytes:
            break

    return b"".join(parts), selected_paths


def measure_concatenated_size(paths, separator, lowercase=False):
    separator_bytes = separator.encode("utf-8")
    total_bytes = 0

    for index, path in enumerate(paths):
        data = read_text_bytes(path, lowercase=lowercase)
        if index > 0:
            total_bytes += len(separator_bytes)
        total_bytes += len(data)

    return total_bytes


def write_bytes(path, data):
    with open(path, "wb") as file:
        file.write(data)


def compression_stats_for_bytes(ppm_cmd, ppm_config, data, temp_dir=None, suffix=".txt"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as temp_in:
        input_path = temp_in.name
        temp_in.write(data)

    output_path = input_path + ".ppm"
    command = [ppm_cmd, "encode", input_path, output_path, *ppm_config.args()]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
        )
        output_bytes = os.path.getsize(output_path)

        match_bits = re.search(r"Bits totais:\s+(\d+)", result.stdout)
        total_bits = int(match_bits.group(1)) if match_bits else output_bytes * 8

        return {
            "compressed_bytes": output_bytes,
            "compressed_bits": total_bits,
        }
    finally:
        remove_artifacts(
            input_path,
            output_path,
            output_path + ".rate.csv",
            output_path + ".reset.csv",
        )


def split_text_into_chunks(text, chunk_size, min_chunk_bytes=1):
    if chunk_size <= 0:
        chunk = text.strip()
        if not chunk:
            return []
        return [chunk.encode("utf-8")]

    words = text.split()
    if not words:
        return []

    chunks = []
    current_words = []
    current_bytes = 0

    for word in words:
        word_bytes = len(word.encode("utf-8"))
        added_bytes = word_bytes + (1 if current_words else 0)

        if current_words and current_bytes + added_bytes > chunk_size:
            chunk = " ".join(current_words)
            chunk_bytes = chunk.encode("utf-8")
            if len(chunk_bytes) >= min_chunk_bytes:
                chunks.append(chunk_bytes)
            current_words = [word]
            current_bytes = word_bytes
        else:
            current_words.append(word)
            current_bytes += added_bytes

    if current_words:
        chunk = " ".join(current_words)
        chunk_bytes = chunk.encode("utf-8")
        if len(chunk_bytes) >= min_chunk_bytes:
            chunks.append(chunk_bytes)

    return chunks
