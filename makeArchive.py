from pathlib import Path # noqa: I001
import subprocess
import sys
import zipfile
from tqdm import tqdm


def main() -> int:
	script = Path(__file__).resolve()
	try:
		root = Path(
			subprocess.check_output(
				["git", "rev-parse", "--show-toplevel"], text=True
			).strip()
		).resolve()
		output = root / f"{root.name}.zip"
		files = subprocess.check_output(
			["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
		).split(b"\0")
		paths = {
			root / Path(raw_path.decode())
			for raw_path in files
			if raw_path
		}
		data_dir = root / "data"
		if data_dir.is_dir():
			paths.update(path for path in data_dir.rglob("*") if path.is_file())
	except (subprocess.CalledProcessError, FileNotFoundError):
		print("Run this script from inside a Git repository.", file=sys.stderr)
		return 1

	with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
		for path in tqdm(paths, desc="Creating archive", unit="file"):
			if path in {script, output} or not path.is_file():
				continue
			archive.write(path, path.relative_to(root))

	print(f"Created {output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
