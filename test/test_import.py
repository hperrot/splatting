from re import sub
import tempfile
import distutils.dir_util
import os
import subprocess


def test_import_from_splatting_root():
    from splatting import Splatting


def test_import_from_other_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        splatting_folder = os.path.dirname(os.path.dirname(__file__))
        tmp_splatting_folder = os.path.join(tmpdir, "src", "splatting")
        files_copied = distutils.dir_util.copy_tree(
            splatting_folder, tmp_splatting_folder
        )
        orig_dir = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = subprocess.run(
                ["python", "-c", "from src.splatting import Splatting"],
                capture_output=True,
            )
            if result.returncode != 0:
                print("stdout:", result.stdout.decode())
                print("stderr:", result.stderr.decode())
                raise ImportError
        finally:
            os.chdir(orig_dir)
        pass
