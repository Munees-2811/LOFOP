"""Tests for dataset statistics."""

from lofop.data.dataset import BoxAnnotation, Category, Dataset, Sample
from lofop.data.statistics import compute_stats


class TestStats:
    def test_counts(self, canonical_dataset):
        result = compute_stats(canonical_dataset)
        assert result.num_images == 2
        assert result.num_annotations == 3
        assert result.num_categories == 2
        assert result.per_category == {"cat": 2, "dog": 1}
        assert result.boxes_per_image_mean == 1.5
        assert result.images_without_annotations == 0
        assert result.image_sizes == {(100, 80): 1, (64, 64): 1}

    def test_size_breakdown_follows_coco_convention(self):
        ds = Dataset("t", [Category(1, "cat")])
        ds.add_sample(Sample(
            image="a.png", width=500, height=500,
            annotations=[
                BoxAnnotation(bbox=(0.0, 0.0, 10.0, 10.0), category_id=1),     # 100 px^2 small
                BoxAnnotation(bbox=(0.0, 0.0, 50.0, 50.0), category_id=1),     # 2500 medium
                BoxAnnotation(bbox=(0.0, 0.0, 200.0, 200.0), category_id=1),   # 40000 large
            ],
        ))
        result = compute_stats(ds)
        assert result.size_breakdown == {"small": 1, "medium": 1, "large": 1}

    def test_empty_dataset(self):
        ds = Dataset("t", [Category(1, "cat")])
        result = compute_stats(ds)
        assert result.num_images == 0
        assert result.boxes_per_image_mean == 0.0
        assert result.box_area_mean == 0.0
        assert result.per_category == {"cat": 0}

    def test_markdown_and_dict_render(self, canonical_dataset):
        result = compute_stats(canonical_dataset)
        md = result.to_markdown()
        assert "| cat | 2 |" in md
        assert "| 100x80 | 1 |" in md
        as_dict = result.to_dict()
        assert as_dict["image_sizes"] == {"100x80": 1, "64x64": 1}
