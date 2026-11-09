"""The profile: loading and validation, writing, hashing, and reading resumes."""
import pytest
from docx import Document

from jsearch.config import ROOT
from jsearch.profile import (Profile, ProfileError, load_profile, profile_hash, read_resumes, resume_text,
                             write_profile)


def tiny_pdf(text: str) -> bytes:
    """A minimal valid one-page PDF showing `text` (enough for pypdf's text extraction)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def tiny_docx(path, paragraphs: list[str], cell: str | None = None) -> None:
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    if cell:
        doc.add_table(rows=1, cols=1).cell(0, 0).text = cell
    doc.save(str(path))


def write_yaml(tmp_path, text: str):
    path = tmp_path / "profile.yaml"
    path.write_text(text, encoding="utf-8")
    return path


VALID = """name: T
experience_years: 0
target_roles: [AI Engineer]
skills: {python: 3, rag: 2}
"""


# --- load / validate ---------------------------------------------------------------------------------------

def test_the_example_profile_is_valid():
    p = load_profile(ROOT / "profile.example.yaml")
    assert p.skills["python"] == 3 and [r.label for r in p.resumes] == ["AI", "PM"]


def test_minimal_profile_loads_with_defaults(tmp_path):
    p = load_profile(write_yaml(tmp_path, VALID))
    assert p.aliases == {} and p.resumes == [] and p.summary == ""


def test_missing_profile_is_none(tmp_path):
    assert load_profile(tmp_path / "absent.yaml") is None


@pytest.mark.parametrize("change, message", [
    ("skills: {python: 5}", "weights must be 1, 2 or 3"),
    ("aliases: {java: [jvm]}", "aliases for skills that are not in skills: ['java']"),
    ("resumes: [{label: A}, {label: A}]", "resume labels must be unique"),
    ("resumes: [{label: A, reserve: -1}]", "reserve"),
    ("target_roles: []", "target_roles"),
    ("experience_years: -1", "experience_years"),
    ("hobbies: [chess]", "hobbies: Extra inputs are not permitted"),
])
def test_invalid_profiles_name_the_problem(tmp_path, change, message):
    key = change.split(":")[0]
    text = "\n".join(line for line in VALID.splitlines() if not line.startswith(key + ":")) + "\n" + change
    with pytest.raises(ProfileError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_profile(write_yaml(tmp_path, text))


def test_invalid_yaml_is_a_profile_error(tmp_path):
    with pytest.raises(ProfileError, match="not valid YAML"):
        load_profile(write_yaml(tmp_path, "skills: [python: 3"))


# --- write / hash ------------------------------------------------------------------------------------------

def test_write_never_overwrites(tmp_path):
    p = load_profile(ROOT / "profile.example.yaml")
    target = tmp_path / "profile.yaml"
    assert write_profile(p, target) == target
    second = write_profile(p, target)
    assert second.name == "profile.new.yaml"
    assert load_profile(second) == p                       # round trip
    assert second.read_text(encoding="utf-8").startswith("# Your job-matching profile")


def test_hash_changes_with_the_profile():
    p = load_profile(ROOT / "profile.example.yaml")
    q = p.model_copy(update={"skills": p.skills | {"go": 1}})
    assert profile_hash(p) == profile_hash(p.model_copy()) != profile_hash(q)


# --- resumes -----------------------------------------------------------------------------------------------

def test_docx_text_includes_tables(tmp_path):
    path = tmp_path / "cv.docx"
    tiny_docx(path, ["Asha Example", "Python, RAG"], cell="FastAPI")
    assert resume_text(path) == "Asha Example\nPython, RAG\nFastAPI"


def test_pdf_text(tmp_path):
    path = tmp_path / "cv.pdf"
    path.write_bytes(tiny_pdf("Built a RAG system"))
    assert "Built a RAG system" in resume_text(path)


def test_read_resumes_skips_unreadable_and_other_files(tmp_path, caplog):
    tiny_docx(tmp_path / "good.docx", ["Python developer"])
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "broken.docx").write_bytes(b"not a zip")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert read_resumes(tmp_path) == {"good.docx": "Python developer"}
    assert "skipping broken.pdf" in caplog.text and "skipping broken.docx" in caplog.text


def test_read_resumes_needs_at_least_one(tmp_path):
    with pytest.raises(ProfileError, match="no readable"):
        read_resumes(tmp_path)
    with pytest.raises(ProfileError, match="is not a folder"):
        read_resumes(tmp_path / "nope")
