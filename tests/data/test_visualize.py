"""Tests for dataset visualization (torch-free, Pillow only)."""

from PIL import Image

from lofop.data import draw_boxes, render_sample, visualize_dataset


class TestDrawBoxes:
    def _image(self):
        return Image.new("RGB", (100, 80), color=(10, 10, 10))

    def test_returns_new_image_same_size(self):
        source = self._image()
        out = draw_boxes(source, [(10, 10, 40, 40)])
        assert isinstance(out, Image.Image)
        assert out.size == source.size
        # Source is not mutated (drawing changes pixels; a copy leaves it alone).
        assert source.getpixel((25, 25)) == (10, 10, 10)

    def test_box_changes_pixels(self):
        out = draw_boxes(self._image(), [(10, 10, 40, 40)], width=3)
        # Some pixel on the rectangle outline must differ from the background.
        assert any(out.getpixel((x, 10)) != (10, 10, 10) for x in range(10, 41))

    def test_caption_with_names_and_scores(self):
        # Just exercises the label/score/name path end to end without error.
        out = draw_boxes(
            self._image(), [(5, 20, 60, 70)], labels=[1], scores=[0.9],
            class_names=["cat", "dog"],
        )
        assert out.size == (100, 80)

    def test_handles_no_boxes(self):
        out = draw_boxes(self._image(), [])
        assert out.size == (100, 80)


class TestRenderDataset:
    def test_render_sample(self, canonical_dataset):
        sample = canonical_dataset.samples[0]
        out = render_sample(canonical_dataset, sample)
        assert out.size == (sample.width, sample.height)

    def test_visualize_dataset_writes_files(self, canonical_dataset, tmp_path):
        paths = visualize_dataset(canonical_dataset, tmp_path / "vis")
        assert len(paths) == len(canonical_dataset)
        assert all(p.is_file() and p.suffix == ".png" for p in paths)

    def test_limit(self, canonical_dataset, tmp_path):
        paths = visualize_dataset(canonical_dataset, tmp_path / "vis", limit=1)
        assert len(paths) == 1
