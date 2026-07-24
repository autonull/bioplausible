"""Tests for data curricula package."""

import pytest

from bioplausible.data.curricula import (
    CURRICULA,
    AntiCurriculum,
    CurriculumScheduler,
    FixedCurriculum,
    ProgressiveCurriculum,
)


class TestFixedCurriculum:
    def test_constant_difficulty(self):
        c = FixedCurriculum(0.5)
        for epoch in range(10):
            assert c.get_difficulty(epoch, 10) == 0.5

    def test_default(self):
        c = FixedCurriculum()
        assert c.get_difficulty(0, 10) == 1.0

    def test_description(self):
        c = FixedCurriculum(0.3)
        assert "0.3" in c.description()


class TestProgressiveCurriculum:
    def test_linear_increase(self):
        c = ProgressiveCurriculum(0.0, 1.0)
        assert c.get_difficulty(0, 10) == 0.0
        assert c.get_difficulty(9, 10) == 1.0
        assert c.get_difficulty(5, 10) == pytest.approx(0.5555, rel=0.01)

    def test_single_epoch(self):
        c = ProgressiveCurriculum(0.0, 1.0)
        assert c.get_difficulty(0, 1) == 1.0

    def test_partial_range(self):
        c = ProgressiveCurriculum(0.2, 0.8)
        assert c.get_difficulty(0, 10) == 0.2
        assert c.get_difficulty(9, 10) == 0.8


class TestAntiCurriculum:
    def test_linear_decrease(self):
        c = AntiCurriculum(1.0, 0.0)
        assert c.get_difficulty(0, 10) == 1.0
        assert c.get_difficulty(9, 10) == 0.0

    def test_single_epoch(self):
        c = AntiCurriculum(1.0, 0.0)
        assert c.get_difficulty(0, 1) == 0.0


class TestCurriculumScheduler:
    def test_step(self):
        curriculum = ProgressiveCurriculum(0.0, 1.0)
        scheduler = CurriculumScheduler(curriculum)
        diff = scheduler.step(5, 10)
        assert diff == pytest.approx(0.5555, rel=0.01)
        assert scheduler.current_difficulty == diff

    def test_apply_fn(self):
        values = []

        def track(diff):
            values.append(diff)

        curriculum = ProgressiveCurriculum(0.0, 1.0)
        scheduler = CurriculumScheduler(curriculum, apply_fn=track)
        scheduler.step(0, 5)
        assert values == [0.0]
        scheduler.step(4, 5)
        assert values == [0.0, 1.0]

    def test_description(self):
        c = FixedCurriculum()
        scheduler = CurriculumScheduler(c)
        assert "Fixed" in scheduler.description()


class TestPrebuiltCurricula:
    def test_default_curriculum(self):
        assert "default" in CURRICULA
        assert isinstance(CURRICULA["default"], FixedCurriculum)

    def test_progressive(self):
        assert "progressive" in CURRICULA
        assert isinstance(CURRICULA["progressive"], ProgressiveCurriculum)

    def test_anti(self):
        assert "anti" in CURRICULA
        assert isinstance(CURRICULA["anti"], AntiCurriculum)

    def test_easy_first(self):
        assert "easy_first" in CURRICULA
        c = CURRICULA["easy_first"]
        assert c.get_difficulty(0, 10) == 0.0
        assert c.get_difficulty(9, 10) == 0.5
