"""Tests for the dataset validator."""

from lofop.data.dataset import BoxAnnotation, Category, Dataset, Sample
from lofop.data.validator import Severity, validate_dataset


def make_dataset(annotations, width=100, height=100, image="a.png"):
    ds = Dataset("t", [Category(1, "cat")])
    ds.add_sample(Sample(image=image, width=width, height=height, annotations=annotations))
    return ds


class TestValidator:
    def test_clean_dataset_passes(self, canonical_dataset):
        report = validate_dataset(canonical_dataset)
        assert report.ok
        assert report.issues == []

    def test_degenerate_box(self):
        ds = make_dataset([BoxAnnotation(bbox=(50.0, 50.0, 50.0, 60.0), category_id=1)])
        report = validate_dataset(ds, check_images=False)
        assert not report.ok
        assert "degenerate" in str(report.errors[0])

    def test_out_of_bounds_box(self):
        ds = make_dataset([BoxAnnotation(bbox=(0.0, 0.0, 150.0, 50.0), category_id=1)])
        report = validate_dataset(ds, check_images=False)
        assert any("outside image bounds" in str(e) for e in report.errors)

    def test_bounds_tolerance_allows_slight_overflow(self):
        ds = make_dataset([BoxAnnotation(bbox=(0.0, 0.0, 101.5, 50.0), category_id=1)])
        assert validate_dataset(ds, check_images=False).ok

    def test_non_finite_coordinates(self):
        ds = make_dataset([BoxAnnotation(bbox=(0.0, 0.0, float("nan"), 10.0), category_id=1)])
        report = validate_dataset(ds, check_images=False)
        assert any("non-finite" in str(e) for e in report.errors)

    def test_unknown_category(self):
        ds = make_dataset([BoxAnnotation(bbox=(0.0, 0.0, 10.0, 10.0), category_id=9)])
        report = validate_dataset(ds, check_images=False)
        assert any("unknown category" in str(e) for e in report.errors)

    def test_missing_image_is_warning(self):
        ds = make_dataset([BoxAnnotation(bbox=(0.0, 0.0, 10.0, 10.0), category_id=1)])
        report = validate_dataset(ds, check_images=True)
        assert report.ok  # warnings only
        assert any("not found" in str(w) for w in report.warnings)

    def test_duplicate_image_entries(self):
        ds = Dataset("t", [Category(1, "cat")])
        ds.add_sample(Sample(image="a.png", width=10, height=10))
        ds.add_sample(Sample(image="a.png", width=10, height=10))
        report = validate_dataset(ds, check_images=False)
        assert any("duplicate image" in str(w) for w in report.warnings)

    def test_empty_dataset(self):
        ds = Dataset("t", [])
        report = validate_dataset(ds, check_images=False)
        assert not report.ok
        messages = [i.message for i in report.errors]
        assert any("no categories" in m for m in messages)
        assert any("no samples" in m for m in messages)

    def test_no_annotations_warning(self):
        ds = Dataset("t", [Category(1, "cat")])
        ds.add_sample(Sample(image="a.png", width=10, height=10))
        report = validate_dataset(ds, check_images=False)
        assert any("no annotations" in str(w) for w in report.warnings)

    def test_severity_and_summary(self):
        ds = make_dataset([BoxAnnotation(bbox=(5.0, 5.0, 5.0, 5.0), category_id=1)])
        report = validate_dataset(ds, check_images=False)
        assert report.errors[0].severity is Severity.ERROR
        assert "FAILED" in report.summary()
