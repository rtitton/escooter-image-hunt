import contextlib
import csv
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from dataset_stats import load_dataset, read_annotation, read_images, report, summarize


class DatasetStatsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.names = {0: "person", 1: "escooter", 2: "unused"}
        self.a = self.image("a", "1 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n")
        self.b = self.image("nested/b", "0 0.5 0.5 0.2 0.2\n")
        self.empty = self.image("empty", "")
        self.missing = self.image("missing", None)
        self.data_yaml = self.root / "data.yaml"

    def image(self, name, label):
        image = self.root / "images" / f"{name}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.touch()
        if label is not None:
            path = self.root / "labels" / f"{name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(label)
        return image

    def write_yaml(self, **entries):
        data = {"names": self.names, "nc": 3, **entries}
        self.data_yaml.write_text(yaml.safe_dump(data))
        return load_dataset(self.data_yaml)

    def test_background_and_class_counts(self):
        images = [self.a, self.b, self.empty, self.missing]
        annotations = {image: read_annotation(image, self.names) for image in images}
        stat = summarize(images, annotations)
        self.assertEqual((stat.images, stat.background, stat.missing, stat.empty),
                         (4, 2, 1, 1))
        self.assertEqual(stat.instances, {0: 1, 1: 2})
        self.assertEqual(stat.class_images, {0: 1, 1: 1})
        self.assertEqual(stat.max_instances, 2)

    def test_recursive_directory_and_names_list(self):
        names, splits = self.write_yaml(train="images", names=list(self.names.values()))
        self.assertEqual(names, self.names)
        self.assertEqual(set(splits["train"]), {self.a, self.b, self.empty, self.missing})

    def test_relative_root_and_multiple_sources(self):
        folder = self.root / "subset"
        folder.mkdir()
        manifest = folder / "train.txt"
        manifest.write_text(f"{self.a}\n{self.a}\n\n")
        _, splits = self.write_yaml(path="subset", train=["train.txt", str(self.b.parent)])
        self.assertEqual(splits["train"], [self.a, self.a, self.b])

    def test_relative_manifest_paths_and_cli(self):
        manifest = self.root / "train.txt"
        manifest.write_text("./images/a.jpg\nimages/a.jpg\n./images/empty.jpg\n")
        self.write_yaml(train="train.txt")
        script = Path(__file__).with_name("dataset_stats.py")
        result = subprocess.run([sys.executable, str(script), str(self.data_yaml)],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        total = next(line for line in lines if line.startswith("TOTALE")).split()
        self.assertEqual(total, ["TOTALE", "3", "4", "1", "0", "1", "1.33", "2"])
        classes = [line.split() for line in lines if line.split() and line.split()[0].isdigit()]
        self.assertEqual([row[1] for row in classes], ["escooter", "person", "unused"])
        self.assertEqual(classes[0], ["1", "escooter", "4", "100.00", "2", "4"])
        self.assertNotIn("uniche", result.stdout)
        self.assertNotIn("ripetizioni", result.stdout)

    def test_report_zero_instances(self):
        splits = {"train": [self.empty], "val": [self.empty, self.missing]}
        annotations = {image: read_annotation(image, self.names)
                       for images in splits.values() for image in images}
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            report(self.names, splits, annotations)
        output = stream.getvalue()
        self.assertNotIn("Percorsi condivisi", output)
        self.assertIn("0.00", output)
        self.assertIn("unused", output)

    def test_sorting_and_csv_match_console(self):
        names = {0: "Zebra, grande", 1: "beta", 2: "Alpha"}
        splits = {"train": [self.a], "val": [self.b, self.empty]}
        annotations = {image: read_annotation(image, names)
                       for images in splits.values() for image in images}
        csv_path = self.root / "stats.csv"
        for sort, expected in (("frequency", [1, 0, 2]), ("id", [0, 1, 2]),
                               ("name", [2, 1, 0])):
            with self.subTest(sort=sort):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    report(names, splits, annotations, sort, csv_path)
                console_ids = [int(line.split()[0]) for line in stream.getvalue().splitlines()
                               if line.split() and line.split()[0].isdigit()]
                with csv_path.open(newline="", encoding="utf-8") as file:
                    rows = list(csv.DictReader(file))
                summary = [row for row in rows if row["tipo"] == "riepilogo"]
                classes = [row for row in rows if row["tipo"] == "classe"]
                self.assertEqual(console_ids, expected)
                self.assertEqual([int(row["ID"]) for row in classes], expected)
                self.assertEqual([row["split"] for row in summary], ["train", "val", "TOTALE"])
                self.assertEqual(summary[-1]["immagini"], "3")
                self.assertEqual(summary[-1]["istanze"], "3")
                self.assertEqual(summary[-1]["classe"], "")
                zebra = next(row for row in classes if row["ID"] == "0")
                self.assertEqual(zebra["classe"], "Zebra, grande")
                self.assertEqual((zebra["ist. train"], zebra["ist. val"]), ("0", "1"))
                self.assertEqual(zebra["sfondo"], "")
                self.assertTrue(all(None not in row for row in rows))

    def test_cli_csv_options_and_errors(self):
        self.write_yaml(train="images")
        script = Path(__file__).with_name("dataset_stats.py")
        csv_path = self.root / "stats.csv"
        result = subprocess.run([sys.executable, str(script), str(self.data_yaml),
                                 "--sort", "name", "--csv", str(csv_path)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Riepilogo:", result.stdout)
        self.assertIn(f"CSV: {csv_path}", result.stdout)
        self.assertTrue(csv_path.is_file())
        for arguments in (["--sort", "invalid"],
                          ["--csv", str(self.root / "absent" / "stats.csv")],
                          ["--csv", str(self.data_yaml)]):
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, str(script), str(self.data_yaml),
                                         *arguments], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)
        self.assertEqual(load_dataset(self.data_yaml)[0], self.names)

    def test_image_symlink_keeps_its_own_labels(self):
        overlay = self.root / "filtered"
        (overlay / "labels").mkdir(parents=True)
        (overlay / "images").symlink_to(self.root / "images", target_is_directory=True)
        (overlay / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        images = read_images(overlay / "images")
        image = next(image for image in images if image.name == "a.jpg")
        self.assertEqual(image, overlay / "images" / "a.jpg")
        self.assertEqual(read_annotation(image, self.names).instances, {0: 1})

    def test_invalid_labels_are_not_silently_counted(self):
        label = self.root / "labels" / "a.txt"
        for line in ("4 0.5 0.5 0.2 0.2", "1.5 0.5 0.5 0.2 0.2",
                     "1 nan 0.5 0.2 0.2", "1 nope 0.5 0.2 0.2", "1 0.5"):
            with self.subTest(line=line):
                label.write_text(line)
                with self.assertRaisesRegex(ValueError, "a.txt:1"):
                    read_annotation(self.a, self.names)

    def test_invalid_configuration_and_sources(self):
        for entries in ({"train": "images", "nc": 2},
                        {"train": "images", "names": {1: "escooter"}},
                        {"train": "images", "names": {0.5: "escooter"}},
                        {"train": "images", "path": 10},
                        {"train": [None]}, {}):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.write_yaml(**entries)
        with self.assertRaises(FileNotFoundError):
            self.write_yaml(train="absent.txt")
        (self.root / "empty.txt").touch()
        with self.assertRaisesRegex(ValueError, "nessuna immagine"):
            self.write_yaml(train="empty.txt")

    def test_missing_image_and_broken_label_symlink(self):
        with self.assertRaises(FileNotFoundError):
            read_annotation(self.root / "images" / "absent.jpg", self.names)
        (self.root / "labels" / "missing.txt").symlink_to(self.root / "absent.txt")
        with self.assertRaisesRegex(ValueError, "symlink rotto"):
            read_annotation(self.missing, self.names)


if __name__ == "__main__":
    unittest.main()
