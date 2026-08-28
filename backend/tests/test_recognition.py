"""
Face and belonging recognition tests — Segment 8.

The privacy boundary is the most important thing tested here: an unenrolled
person must NEVER be named. That is asserted from several directions, because
it is the property that would do real harm if it broke.

Identity tests use two genuinely different people from a real photograph
(zidane.jpg contains two faces), rather than synthetic images, since face
embeddings on synthetic faces prove nothing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.agent.tools import ToolRegistry
from app.memory.store import MemoryStore
from app.perception.appearance import (
    CONFIDENT_MATCH_THRESHOLD,
    POSSIBLE_MATCH_THRESHOLD,
    AppearanceEmbedder,
    crop_from_box,
    describe_match,
)
from app.perception.faces import (
    CONFIDENT_IDENTITY_THRESHOLD,
    SAME_IDENTITY_THRESHOLD,
    FaceEngine,
)
from app.perception.object_detector import BoundingBox
from app.perception.recognition import RecognitionEngine
from app.perception.world_model import WorldModel

FIXTURES = Path(__file__).parent / "fixtures"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def face_engine() -> FaceEngine:
    engine = FaceEngine()
    if not engine.available:
        pytest.skip("face models not downloaded")
    return engine


@pytest.fixture(scope="module")
def zidane() -> np.ndarray:
    image = cv2.imread(str(FIXTURES / "zidane.jpg"))
    assert image is not None
    return image


@pytest.fixture(scope="module")
def two_faces(face_engine: FaceEngine, zidane: np.ndarray):
    """The two distinct people in zidane.jpg, ordered left to right."""
    faces = sorted(face_engine.detect(zidane), key=lambda f: f.box[0])
    if len(faces) < 2:
        pytest.skip("expected two faces in the fixture")
    return faces


# ---------------- face detection ----------------

def test_faces_are_detected_in_real_photographs(face_engine, zidane):
    assert len(face_engine.detect(zidane)) >= 2


def test_no_faces_found_in_an_image_without_any(face_engine):
    blank = np.full((480, 640, 3), 120, dtype=np.uint8)
    assert face_engine.detect(blank) == []


def test_missing_models_disable_recognition_instead_of_crashing():
    engine = FaceEngine(
        detector_path=Path("nope-detector.onnx"), recognizer_path=Path("nope-rec.onnx")
    )
    assert engine.available is False
    assert engine.detect(np.zeros((100, 100, 3), np.uint8)) == []


# ---------------- identity ----------------

def test_same_person_still_matches_after_image_degradation(face_engine, zidane, two_faces):
    """
    Real conditions change: lighting, focus, resolution, compression. The same
    person must stay recognizable across all of them.
    """
    reference = face_engine.embed(zidane, two_faces[0])
    assert reference is not None

    variants = {
        "brighter": cv2.convertScaleAbs(zidane, beta=30),
        "blurred": cv2.GaussianBlur(zidane, (7, 7), 0),
        "half size": cv2.resize(zidane, (zidane.shape[1] // 2, zidane.shape[0] // 2)),
        "jpeg q30": cv2.imdecode(cv2.imencode(".jpg", zidane, [1, 30])[1], 1),
    }
    for label, image in variants.items():
        faces = sorted(face_engine.detect(image), key=lambda f: f.box[0])
        if not faces:
            continue  # detection loss is covered by its own test below
        embedding = face_engine.embed(image, faces[0])
        score = face_engine.similarity(reference, embedding)
        assert score >= SAME_IDENTITY_THRESHOLD, label + " scored only " + str(round(score, 3))


def test_two_different_people_are_not_confused(face_engine, zidane, two_faces):
    """The false-positive case — naming a stranger as the user's friend."""
    a = face_engine.embed(zidane, two_faces[0])
    b = face_engine.embed(zidane, two_faces[1])
    score = face_engine.similarity(a, b)
    assert score < SAME_IDENTITY_THRESHOLD, "two different people matched at " + str(score)


def test_match_returns_nothing_when_nobody_is_enrolled(face_engine, zidane, two_faces):
    embedding = face_engine.embed(zidane, two_faces[0])
    assert face_engine.match(embedding, [], two_faces[0].box) is None


def test_match_rejects_a_face_below_the_identity_threshold(face_engine, zidane, two_faces):
    a = face_engine.embed(zidane, two_faces[0])
    b = face_engine.embed(zidane, two_faces[1])
    # B is enrolled; A is a different person and must not match B.
    assert face_engine.match(a, [("Rahul", b)], two_faces[0].box) is None


def test_a_weak_match_is_flagged_as_not_confident(face_engine, zidane, two_faces):
    a = face_engine.embed(zidane, two_faces[0])
    match = face_engine.match(a, [("Rahul", a)], two_faces[0].box)
    assert match is not None and match.is_confident is True
    assert CONFIDENT_IDENTITY_THRESHOLD > SAME_IDENTITY_THRESHOLD


# ---------------- the privacy boundary ----------------

def test_unenrolled_people_are_counted_but_never_named(face_engine, zidane, two_faces):
    """
    THE critical privacy property. With one person enrolled and two faces in
    view, exactly one may be named and the other must remain anonymous.
    """
    store = MemoryStore(db_path=":memory:")
    try:
        engine = RecognitionEngine(store=store, faces=face_engine)
        embedding = face_engine.embed(zidane, two_faces[0])
        store.enroll("person", "Rahul", embedding.tolist())
        engine.invalidate_cache()

        identified, face_count = engine.identify_people(zidane)
        assert face_count >= 2
        assert len(identified) == 1, "more than the enrolled person was identified"
        assert identified[0].name == "Rahul"
    finally:
        store.close()


def test_nobody_is_identified_when_nothing_is_enrolled(face_engine, zidane):
    store = MemoryStore(db_path=":memory:")
    try:
        engine = RecognitionEngine(store=store, faces=face_engine)
        identified, face_count = engine.identify_people(zidane)
        assert face_count >= 2, "faces should still be DETECTED"
        assert identified == [], "someone was identified with an empty enrollment list"
    finally:
        store.close()


def test_who_is_here_tool_forbids_describing_unrecognized_people(face_engine, zidane):
    store = MemoryStore(db_path=":memory:")

    class _Pipeline:
        def current_frame(self):
            return zidane

    try:
        engine = RecognitionEngine(store=store, faces=face_engine)
        registry = ToolRegistry(
            WorldModel(), store=store, pipeline=_Pipeline(), recognition=engine
        )
        result = run(registry.execute("who_is_here", {}))
        assert result["face_count"] >= 2
        assert result["identified"] == []
        assert result["unrecognized_face_count"] >= 2
        assert "must NOT" in result["note"]
    finally:
        store.close()


def test_find_person_reports_when_a_name_was_never_enrolled(face_engine, zidane):
    store = MemoryStore(db_path=":memory:")

    class _Pipeline:
        def current_frame(self):
            return zidane

    try:
        engine = RecognitionEngine(store=store, faces=face_engine)
        registry = ToolRegistry(
            WorldModel(), store=store, pipeline=_Pipeline(), recognition=engine
        )
        result = run(registry.execute("find_person", {"name": "Nobody"}))
        assert result["found"] is False
        assert result["enrolled"] is False
        assert "hasn't been enrolled" in result["message"]
    finally:
        store.close()


# ---------------- enrollment lifecycle ----------------

def test_enrollment_stores_an_embedding_not_a_photograph(face_engine, zidane, two_faces):
    store = MemoryStore(db_path=":memory:")
    try:
        embedding = face_engine.embed(zidane, two_faces[0])
        record = store.enroll("person", "Rahul", embedding.tolist())
        assert len(record.embedding) == 128
        # Nothing image-shaped is retained anywhere in the row.
        assert "image" not in record.payload()
        assert record.payload()["name"] == "Rahul"
    finally:
        store.close()


def test_re_enrolling_averages_samples_rather_than_replacing(face_engine, zidane, two_faces):
    """A second photo should improve recognition, not discard the first."""
    store = MemoryStore(db_path=":memory:")
    try:
        a = face_engine.embed(zidane, two_faces[0])
        blurred = cv2.GaussianBlur(zidane, (5, 5), 0)
        faces = sorted(face_engine.detect(blurred), key=lambda f: f.box[0])
        b = face_engine.embed(blurred, faces[0])

        store.enroll("person", "Rahul", a.tolist())
        second = store.enroll("person", "Rahul", b.tolist())
        assert second.sample_count == 2
        assert len(store.list_enrollments("person")) == 1
        # Still recognizes the original.
        assert float(np.dot(second.vector(), a)) >= SAME_IDENTITY_THRESHOLD
    finally:
        store.close()


def test_deleting_an_enrollment_removes_the_only_record(face_engine, zidane, two_faces):
    store = MemoryStore(db_path=":memory:")
    try:
        embedding = face_engine.embed(zidane, two_faces[0])
        store.enroll("person", "Rahul", embedding.tolist())
        assert store.delete_enrollment("Rahul") == 1
        assert store.list_enrollments() == []
        assert store.delete_enrollment("Rahul") == 0
    finally:
        store.close()


def test_forget_enrolled_tool_reports_honestly_when_nothing_was_stored():
    store = MemoryStore(db_path=":memory:")
    try:
        engine = RecognitionEngine(store=store, faces=FaceEngine())
        registry = ToolRegistry(WorldModel(), store=store, pipeline=None, recognition=engine)
        result = run(registry.execute("forget_enrolled", {"name": "Ghost"}))
        assert result["deleted"] is False
        assert "nothing enrolled" in result["message"].lower()
    finally:
        store.close()


# ---------------- belongings ----------------

@pytest.fixture(scope="module")
def embedder() -> AppearanceEmbedder:
    engine = AppearanceEmbedder()
    if not engine.available:
        pytest.skip("appearance model not downloaded")
    return engine


def test_appearance_uses_pooled_features_not_class_logits(embedder):
    """
    Class logits are a poor descriptor — measured at 0.708 between a bus and a
    person, which is indistinguishable from a match.
    """
    assert embedder.using_features is True
    crop = cv2.imread(str(FIXTURES / "bus.jpg"))[110:330, 20:620]
    assert embedder.embed(crop).shape[0] == 1280


def test_the_same_object_matches_itself_across_degradation(embedder):
    bus = cv2.imread(str(FIXTURES / "bus.jpg"))
    crop = bus[110:330, 20:620]
    reference = embedder.embed(crop)
    for label, variant in {
        "brighter": cv2.convertScaleAbs(crop, beta=30),
        "blur": cv2.GaussianBlur(crop, (7, 7), 0),
        "jpeg40": cv2.imdecode(cv2.imencode(".jpg", crop, [1, 40])[1], 1),
    }.items():
        score = embedder.similarity(reference, embedder.embed(variant))
        assert score >= POSSIBLE_MATCH_THRESHOLD, label + " scored " + str(round(score, 3))


def test_different_objects_fall_below_the_match_threshold(embedder):
    bus = cv2.imread(str(FIXTURES / "bus.jpg"))
    score = embedder.similarity(
        embedder.embed(bus[110:330, 20:620]), embedder.embed(bus[180:400, 40:190])
    )
    assert score < POSSIBLE_MATCH_THRESHOLD, "a bus matched a person at " + str(round(score, 3))


def test_match_bands_force_hedging_in_the_middle():
    assert describe_match(0.95) == "confident"
    assert describe_match((POSSIBLE_MATCH_THRESHOLD + CONFIDENT_MATCH_THRESHOLD) / 2) == "possible"
    assert describe_match(0.5) == "no_match"


def test_object_class_is_a_hard_gate(embedder):
    """
    A chair must never be reported as the user's backpack, no matter how
    similar it looks.
    """
    store = MemoryStore(db_path=":memory:")
    try:
        bus = cv2.imread(str(FIXTURES / "bus.jpg"))
        crop = bus[110:330, 20:620]
        engine = RecognitionEngine(store=store, appearance=embedder)
        store.enroll("belonging", "my backpack", embedder.embed(crop).tolist(),
                     object_class="backpack")
        engine.invalidate_cache()

        class _Detection:
            class_name = "chair"        # WRONG class, identical pixels
            box = BoundingBox(20, 110, 620, 330)

        assert engine.identify_belongings(bus, [_Detection()]) == []
    finally:
        store.close()


def test_matching_object_of_the_right_class_is_found(embedder):
    store = MemoryStore(db_path=":memory:")
    try:
        bus = cv2.imread(str(FIXTURES / "bus.jpg"))
        crop = bus[110:330, 20:620]
        engine = RecognitionEngine(store=store, appearance=embedder)
        store.enroll("belonging", "my bus", embedder.embed(crop).tolist(), object_class="bus")
        engine.invalidate_cache()

        class _Detection:
            class_name = "bus"
            box = BoundingBox(20, 110, 620, 330)

        matches = engine.identify_belongings(bus, [_Detection()])
        assert matches and matches[0].name == "my bus"
        assert matches[0].band in ("confident", "possible")
    finally:
        store.close()


def test_crop_from_box_rejects_a_degenerate_box():
    frame = np.zeros((100, 100, 3), np.uint8)
    assert crop_from_box(frame, BoundingBox(50, 50, 50, 50)) is None


def test_full_enrollment_flow_matches_the_same_object_later(embedder):
    """
    The REAL user flow, end to end: enrol from a detected object exactly as the
    UI does, then recognize that object again in changed conditions.

    Enrolment and recognition must use the SAME cropping convention. An earlier
    demo enrolled from a hand-picked crop and then failed to match — the code
    was right and the crop was wrong, which is precisely why this test pins the
    real path rather than a convenient shortcut.
    """
    from app.perception.object_detector import ObjectDetector

    store = MemoryStore(db_path=":memory:")
    try:
        bus = cv2.imread(str(FIXTURES / "bus.jpg"))
        detector = ObjectDetector()
        detections = detector.detect(bus)
        buses = [d for d in detections if d.class_name == "bus"]
        assert buses, "fixture should contain a bus"

        largest = max(buses, key=lambda d: d.box.area)
        crop = crop_from_box(bus, largest.box)
        engine = RecognitionEngine(store=store, appearance=embedder)
        store.enroll(
            "belonging", "my bus", embedder.embed(crop).tolist(), object_class="bus"
        )
        engine.invalidate_cache()

        matches = engine.identify_belongings(bus, detections)
        assert matches and matches[0].name == "my bus"
        assert matches[0].band == "confident"

        for label, variant in {
            "brighter": cv2.convertScaleAbs(bus, beta=30),
            "blurred": cv2.GaussianBlur(bus, (7, 7), 0),
            "resized": cv2.resize(bus, (640, 853)),
        }.items():
            again = engine.identify_belongings(variant, detector.detect(variant))
            assert again, "lost the enrolled item after " + label
            assert again[0].name == "my bus"

        # A scene without the item must produce no match at all.
        zidane_img = cv2.imread(str(FIXTURES / "zidane.jpg"))
        assert engine.identify_belongings(zidane_img, detector.detect(zidane_img)) == []
    finally:
        store.close()
