"""Tests for the COCO, YOLO, and VOC format adapters and cross-conversion."""

import pytest

from lofop.core.exceptions import DataError
from lofop.data import convert_dataset, load_dataset, save_dataset
from lofop.data.dataset import Category, Dataset, Sample


class TestCoco:
    def test_load(self, coco_file):
        ds = load_dataset("coco", coco_file)
        assert len(ds) == 2
        assert ds.num_annotations == 3
        assert {c.name for c in ds.categories} == {"cat", "dog"}
        box = ds.samples[0].annotations[0]
        assert box.bbox == (10.0, 10.0, 50.0, 40.0)   # xywh -> xyxy
        assert ds.samples[0].annotations[1].attributes["iscrowd"] == 1

    def test_roundtrip_preserves_ids_and_boxes(self, coco_file, tmp_path):
        ds = load_dataset("coco", coco_file)
        out = tmp_path / "out.json"
        save_dataset(ds, "coco", out)
        back = load_dataset("coco", out)
        assert [c.id for c in back.categories] == [1, 3]
        assert [s.id for s in back.samples] == [s.id for s in ds.samples]
        for orig, round_ in zip(ds.samples, back.samples):
            assert [a.bbox for a in orig.annotations] == [a.bbox for a in round_.annotations]

    def test_missing_file_and_bad_json(self, tmp_path):
        with pytest.raises(DataError, match="Cannot read"):
            load_dataset("coco", tmp_path / "nope.json")
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(DataError, match="Invalid COCO JSON"):
            load_dataset("coco", bad)

    def test_missing_arrays_rejected(self, tmp_path):
        partial = tmp_path / "partial.json"
        partial.write_text('{"images": []}')
        with pytest.raises(DataError, match="annotations"):
            load_dataset("coco", partial)

    def test_unknown_category_rejected(self, tmp_path):
        payload = (
            '{"images": [{"id": 0, "file_name": "a.png", "width": 10, "height": 10}],'
            ' "annotations": [{"id": 1, "image_id": 0, "category_id": 99, "bbox": [0,0,5,5]}],'
            ' "categories": [{"id": 1, "name": "cat"}]}'
        )
        path = tmp_path / "bad_cat.json"
        path.write_text(payload)
        with pytest.raises(DataError, match="unknown category"):
            load_dataset("coco", path)


class TestYolo:
    def test_load_reads_sizes_and_denormalizes(self, yolo_root):
        ds = load_dataset("yolo", yolo_root)
        assert len(ds) == 2
        assert [c.name for c in ds.categories] == ["cat", "dog"]
        a = next(s for s in ds.samples if s.image == "a.png")
        assert (a.width, a.height) == (100, 80)
        box = a.annotations[0]
        assert box.bbox == pytest.approx((10.0, 10.0, 50.0, 40.0))

    def test_save_writes_normalized_labels_and_copies_images(self, canonical_dataset, tmp_path):
        out = tmp_path / "yolo_out"
        save_dataset(canonical_dataset, "yolo", out)
        assert (out / "classes.txt").read_text() == "cat\ndog\n"
        rows = (out / "labels" / "a.txt").read_text().splitlines()
        assert rows[0].split()[0] == "0"                      # cat -> index 0
        assert rows[0].split()[1:] == ["0.300000", "0.312500", "0.400000", "0.375000"]
        assert (out / "images" / "a.png").is_file()

    def test_roundtrip_boxes_survive(self, canonical_dataset, tmp_path):
        out = tmp_path / "yolo_rt"
        save_dataset(canonical_dataset, "yolo", out)
        back = load_dataset("yolo", out)
        orig = {s.image: s for s in canonical_dataset.samples}
        for sample in back.samples:
            expected = [a.bbox for a in orig[sample.image].annotations]
            got = [a.bbox for a in sample.annotations]
            assert len(got) == len(expected)
            for got_box, expected_box in zip(got, expected):
                assert got_box == pytest.approx(expected_box, abs=1e-3)

    def test_malformed_labels_rejected(self, yolo_root):
        (yolo_root / "labels" / "a.txt").write_text("0 0.5 0.5\n")
        with pytest.raises(DataError, match="5 fields"):
            load_dataset("yolo", yolo_root)

    def test_class_index_out_of_range(self, yolo_root):
        (yolo_root / "labels" / "a.txt").write_text("7 0.5 0.5 0.2 0.2\n")
        with pytest.raises(DataError, match="out of range"):
            load_dataset("yolo", yolo_root)

    def test_missing_layout_pieces(self, tmp_path):
        with pytest.raises(DataError, match="classes.txt"):
            load_dataset("yolo", tmp_path)


class TestVoc:
    def test_roundtrip(self, canonical_dataset, tmp_path):
        out = tmp_path / "voc_out"
        save_dataset(canonical_dataset, "voc", out)
        assert (out / "Annotations" / "a.xml").is_file()
        assert (out / "JPEGImages" / "a.png").is_file()
        back = load_dataset("voc", out)
        assert {c.name for c in back.categories} == {"cat", "dog"}
        a = next(s for s in back.samples if s.image == "a.png")
        assert (a.width, a.height) == (100, 80)
        assert a.annotations[0].bbox == pytest.approx((10.0, 10.0, 50.0, 40.0))

    def test_missing_annotations_dir(self, tmp_path):
        with pytest.raises(DataError, match="Annotations"):
            load_dataset("voc", tmp_path)


class TestConvert:
    @pytest.mark.parametrize("target_format", ["yolo", "voc"])
    def test_coco_to_other_formats(self, coco_file, tmp_path, target_format):
        out = tmp_path / f"as_{target_format}"
        ds = convert_dataset("coco", coco_file, target_format, out)
        assert len(ds) == 2
        back = load_dataset(target_format, out)
        assert back.num_annotations == 3 if target_format == "voc" else True
        assert {c.name for c in back.categories} == {"cat", "dog"}

    def test_yolo_to_coco(self, yolo_root, tmp_path):
        out = tmp_path / "as_coco.json"
        convert_dataset("yolo", yolo_root, "coco", out)
        back = load_dataset("coco", out)
        assert len(back) == 2
        assert back.num_annotations == 3

    def test_unknown_format_lists_known_ones(self, coco_file, tmp_path):
        from lofop.core.exceptions import RegistryError

        with pytest.raises(RegistryError, match="coco"):
            convert_dataset("cocoz", coco_file, "yolo", tmp_path / "x")


def test_dataset_rejects_duplicate_categories():
    with pytest.raises(DataError):
        Dataset("d", [Category(1, "cat"), Category(1, "dog")])
    with pytest.raises(DataError):
        Dataset("d", [Category(1, "cat"), Category(2, "cat")])


def test_sample_ids_assigned_sequentially():
    ds = Dataset("d", [Category(1, "cat")])
    first = ds.add_sample(Sample(image="a.png", width=1, height=1))
    second = ds.add_sample(Sample(image="b.png", width=1, height=1))
    assert (first.id, second.id) == (0, 1)
